#include "global.h"
#include "main.h"
#include "battle.h"
#include "battle_eval.h"
#include "battle_main.h"
#include "battle_util.h"
#include "data.h"
#include "event_data.h"
#include "random.h"
#include "item.h"
#include "malloc.h"
#include "pokedex.h"
#include "pokemon.h"
#include "string_util.h"
#include "util.h"
#include "constants/abilities.h"
#include "constants/battle_move_effects.h"
#include "constants/hold_effects.h"
#include "constants/items.h"
#include "constants/moves.h"
#include "constants/species.h"
#include "constants/flags.h"
#include "constants/opponents.h"
#include "constants/trainers.h"

// Battle evaluation engine.
//
// Each turn, once every battler has chosen an action, the engine looks at the
// options each side had and simulates one turn for every combination of them,
// followed by a two-turn greedy rollout, then scores the resulting position as
// a probability that the player wins. The opponent is modelled as playing an
// approximate equilibrium strategy (found by regret matching), and each of the
// player's choices is graded by how much win probability it gave up against
// the best option available.
//
// The simulation is "soft": instead of branching on every random event, each
// Pokemon carries the probability that it is still standing and its expected
// HP if it is. This keeps a whole turn to a few hundred operations, which is
// what makes searching every option feasible on the GBA.

#define Q_ONE   4096
#define Q_HALF  2048
#define QM(a, b) (((s32)(a) * (s32)(b)) >> 12)

#define EVAL_MONS           (PARTY_SIZE * NUM_BATTLE_SIDES)
#define NO_MON              0xFF
#define NO_TARGET           0xFF
#define MAX_OPTIONS         16
#define PRUNED_OPTIONS      3
#define MAX_JOINT_PLAYER    16
#define MAX_JOINT_OPPONENT  16
#define EQUILIBRIUM_ITERS   48
#define STAGE_UNIT          16
#define MAX_STAGE           (6 * STAGE_UNIT)

// Material weights: a healthy Pokemon is worth one unit (Q_ONE).
#define W_ALIVE 1638 // 0.4 for still being able to battle
#define W_HP    2458 // 0.6 scaled by remaining HP

enum
{
    ACT_NONE,
    ACT_MOVE,
    ACT_SWITCH,
};

enum
{
    EST_NONE,
    EST_SLEEP,
    EST_POISON,
    EST_TOXIC,
    EST_BURN,
    EST_FREEZE,
    EST_PARALYSIS,
};

enum
{
    EW_NONE,
    EW_RAIN,
    EW_SUN,
    EW_SAND,
    EW_HAIL,
};

#define EMF_SUBSTITUTE  (1 << 0)
#define EMF_SEEDED      (1 << 1)
#define EMF_NO_ITEM     (1 << 2)
#define EMF_FIRST_TURN  (1 << 3)
#define EMF_TAUNTED     (1 << 4)
#define EMF_CURSED      (1 << 5)
#define EMF_FOCUS       (1 << 6)
#define EMF_LOAFING     (1 << 7)

struct EvalAction
{
    u8 kind;
    u8 index;  // move slot or party slot
    u8 target; // battler, or NO_TARGET
};

// What doesn't change during a simulated turn
struct EvalMon
{
    u16 species;
    u16 maxHP;
    u16 stats[NUM_STATS];
    u16 moves[MAX_MON_MOVES];
    u32 hpRecip;     // (Q_ONE << 16) / (maxHP * 16), to turn HP into a fraction without dividing
    u8 bestSlot[EVAL_MONS]; // strongest attack against each Pokemon, for rollouts
    u8 types[2];
    u8 ability;
    u8 holdEffect;
    u8 holdEffectParam;
    u8 friendship;
    u8 level;
    u8 present;
};

struct EvalMonState
{
    s16 hp;          // expected HP if still standing, in 1/16 HP
    u16 alive;       // probability of still standing
    u16 statusProb;  // probability of having `status`
    u16 confusion;   // probability of being confused
    u8 status;
    u8 statusTurns;  // sleep turns left, or toxic counter
    u8 flags;
    u8 subHP;
    s8 stages[NUM_BATTLE_STATS]; // in 1/16 stages
};

struct EvalState
{
    struct EvalMonState mons[EVAL_MONS];
    u8 active[MAX_BATTLERS_COUNT];
    u8 reflect[NUM_BATTLE_SIDES];
    u8 lightScreen[NUM_BATTLE_SIDES];
    u8 safeguard[NUM_BATTLE_SIDES];
    u8 spikes[NUM_BATTLE_SIDES];
    u8 sleepClause[NUM_BATTLE_SIDES];
    u8 weather;
    u8 wishTurns[MAX_BATTLERS_COUNT];
    s16 wishHP[MAX_BATTLERS_COUNT];
};

// Things that only last for the turn being simulated
struct EvalTurn
{
    u16 protect[MAX_BATTLERS_COUNT];
    u16 endure[MAX_BATTLERS_COUNT];
    u16 flinch[MAX_BATTLERS_COUNT];
    u16 hit[MAX_BATTLERS_COUNT];
    u16 helping[MAX_BATTLERS_COUNT];
    u16 powerBoost[MAX_BATTLERS_COUNT];
    s16 physTaken[MAX_BATTLERS_COUNT];
    s16 specTaken[MAX_BATTLERS_COUNT];
    u8 physFrom[MAX_BATTLERS_COUNT];
    u8 specFrom[MAX_BATTLERS_COUNT];
    u8 done[MAX_BATTLERS_COUNT];
    u8 followMe[NUM_BATTLE_SIDES];
    u16 followMeChance[NUM_BATTLE_SIDES];
};

struct EvalReport
{
    u16 decisions;
    u32 accuracySum;
    u16 foeDecisions;
    u32 foeAccuracySum;
    u16 counts[EVAL_GRADE_COUNT];
    u16 worstLoss;
    u8 worstTurn;
    u16 worstSpecies;
    struct EvalAction worstChosen;
    struct EvalAction worstBest;
    u16 worstChosenMove;
    u16 worstBestMove;
    u16 worstChosenSwitch;
    u16 worstBestSwitch;
};

EWRAM_DATA u8 gEvalReportReady = 0;
EWRAM_DATA u8 gEvalReportText[320] = {0};

static EWRAM_DATA struct EvalReport sReport = {0};

// Working memory, allocated from the heap only while a turn is being graded
struct EvalWork
{
    struct EvalMon mons[EVAL_MONS];
    u16 damage[EVAL_MONS][MAX_MON_MOVES][EVAL_MONS];
    u8 fixedDamage[EVAL_MONS];
    u8 typeChart[NUMBER_OF_MON_TYPES][NUMBER_OF_MON_TYPES];
    struct EvalState root;
    struct EvalState sim;
    struct EvalTurn turn;
    struct EvalAction options[MAX_BATTLERS_COUNT][MAX_OPTIONS];
    u8 numOptions[MAX_BATTLERS_COUNT];
    u8 forced[MAX_BATTLERS_COUNT];
    struct EvalAction chosen[MAX_BATTLERS_COUNT];
    struct EvalAction defaults[MAX_BATTLERS_COUNT];
    u8 pruned[MAX_BATTLERS_COUNT][MAX_OPTIONS];
    u8 numPruned[MAX_BATTLERS_COUNT];
    u16 matrix[MAX_JOINT_PLAYER][MAX_JOINT_OPPONENT];
    u8 jointPlayer[MAX_JOINT_PLAYER][2];
    u8 jointOpponent[MAX_JOINT_OPPONENT][2];
    u16 oppStrategy[MAX_JOINT_OPPONENT];
    u16 protectChance[MAX_BATTLERS_COUNT];
    u16 values[MAX_BATTLERS_COUNT][MAX_OPTIONS];
    u32 valueSums[MAX_OPTIONS];
    u32 valueWeights[MAX_OPTIONS];
    u8 isDouble;
    u8 turnNumber;
    u8 phase;
    u8 stepBattler;
    u8 stepIndex;
    u8 stepOpp;
    u8 numJointPlayer;
    u8 numJointOpponent;
    u8 simStage;
    s32 simMaterial;
    struct EvalAction simActs[MAX_BATTLERS_COUNT];
    u16 playerStrategy[MAX_JOINT_PLAYER];
    struct EvalAction aiAction[MAX_BATTLERS_COUNT];
    u8 aiSkill;
    u8 useAI:1;
    u8 aiReady:1;
    u8 choicesKnown:1;
};

static struct EvalWork *sWork; // in IWRAM, as it is read constantly

#define sMons           (sWork->mons)
#define sDamage         (sWork->damage)
#define sFixedDamage    (sWork->fixedDamage)
#define sTypeChart      (sWork->typeChart)
#define sRoot           (sWork->root)
#define sSim            (sWork->sim)
#define sTurn           (sWork->turn)
#define sOptions        (sWork->options)
#define sNumOptions     (sWork->numOptions)
#define sForced         (sWork->forced)
#define sChosen         (sWork->chosen)
#define sDefault        (sWork->defaults)
#define sPruned         (sWork->pruned)
#define sNumPruned      (sWork->numPruned)
#define sMatrix         (sWork->matrix)
#define sJointPlayer    (sWork->jointPlayer)
#define sJointOpponent  (sWork->jointOpponent)
#define sOppStrategy    (sWork->oppStrategy)
#define sPlayerStrategy (sWork->playerStrategy)
#define sIsDouble       (sWork->isDouble)

// gStatStageRatios as Q12 multipliers, for stages -6 to +6
static const u16 sStageMult[13] = {1024, 1170, 1365, 1638, 2048, 2731, 4096, 6144, 8192, 10240, 12288, 14336, 16384};
static const u16 sAccStageMult[13] = {1352, 1475, 1761, 2048, 2458, 3072, 4096, 5448, 6799, 8192, 9544, 10895, 12288};
static const u16 sCritChance[5] = {256, 512, 1024, 1365, 2048};
static const u16 sSpikesDamage[4] = {0, 512, 683, 1024}; // fraction of max HP, Q12

// Win probability for a material lead of i/16 Pokemon: 1 / (1 + e^(-x / 0.6))
static const u16 sLogistic[129] =
{
    2048, 2155, 2261, 2365, 2469, 2570, 2668, 2763, 2855, 2943, 3028, 3108, 3184, 3256, 3323, 3386,
    3445, 3500, 3551, 3599, 3642, 3683, 3720, 3754, 3785, 3814, 3840, 3864, 3886, 3906, 3924, 3940,
    3955, 3968, 3981, 3992, 4002, 4011, 4019, 4027, 4033, 4040, 4045, 4050, 4055, 4059, 4062, 4066,
    4069, 4071, 4074, 4076, 4078, 4080, 4081, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4090,
    4091, 4091, 4092, 4092, 4093, 4093, 4093, 4093, 4094, 4094, 4094, 4094, 4095, 4095, 4095, 4095,
    4095, 4095, 4095, 4095, 4095, 4095, 4095, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096,
    4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096,
    4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096,
    4096,
};

// Accuracy of a decision that gave up N% win probability (Lichess's formula, after a 1% allowance)
static const u8 sDecisionAccuracy[101] =
{
    100, 100, 96, 91, 87, 84, 80, 76, 73, 70, 67, 64, 61, 58, 55, 53, 51, 48, 46, 44,
    42, 40, 38, 36, 35, 33, 32, 30, 29, 27, 26, 25, 24, 22, 21, 20, 19, 18, 17, 17,
    16, 15, 14, 13, 13, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5,
    5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,
};

// Win probability given up (in tenths of a percent) at the top of each grade
static const u16 sGradeThresholds[EVAL_GRADE_COUNT - 1] = {10, 40, 100, 200};

static const u8 sText_DecisionAccuracy[] = _("Decision accuracy: ");
static const u8 sText_Best[] = _("%\nBest ");
static const u8 sText_Good[] = _(", good ");
static const u8 sText_Inaccurate[] = _(", inaccurate ");
static const u8 sText_Mistakes[] = _("\pMistakes ");
static const u8 sText_Blunders[] = _(", blunders ");
static const u8 sText_Percent[] = _("%");
static const u8 sText_FoeAccuracy[] = _("\nFoe accuracy: ");
static const u8 sText_BiggestSlip[] = _("\pBiggest slip, turn ");
static const u8 sText_Colon[] = _(":\n");
static const u8 sText_Used[] = _(" used ");
static const u8 sText_SwitchedTo[] = _(" swapped to ");
static const u8 sText_Comma[] = _(";");
static const u8 sText_Using[] = _("\pusing ");
static const u8 sText_SwitchingTo[] = _("\pswitching to ");
static const u8 sText_WasBetter[] = _("\nwas the stronger play.");
static const u8 sText_HoldingBack[] = _("\pholding back");

#include "data/trainer_ai_skill.h"

#ifdef EVAL_DEBUG
#define REG_MGBA_DEBUG_ENABLE (*(vu16 *)0x4FFF780)
#define REG_MGBA_DEBUG_FLAGS  (*(vu16 *)0x4FFF700)
#define MGBA_DEBUG_STRING     ((char *)0x4FFF600)

static char *AppendInt(char *s, s32 v)
{
    char tmp[12];
    s32 n = 0;

    if (v < 0)
    {
        *s++ = '-';
        v = -v;
    }
    do
    {
        tmp[n++] = '0' + v % 10;
        v /= 10;
    } while (v != 0);
    while (n > 0)
        *s++ = tmp[--n];
    return s;
}

static void EvalLog(const char *label, s32 a, s32 b, s32 c, s32 d)
{
    char *s = MGBA_DEBUG_STRING;

    REG_MGBA_DEBUG_ENABLE = 0xC0DE;
    while (*label)
        *s++ = *label++;
    *s++ = ' ';
    s = AppendInt(s, a);
    *s++ = ' ';
    s = AppendInt(s, b);
    *s++ = ' ';
    s = AppendInt(s, c);
    *s++ = ' ';
    s = AppendInt(s, d);
    *s = 0;
    REG_MGBA_DEBUG_FLAGS = 3 | 0x100;
}
#else
#define EvalLog(label, a, b, c, d)
#endif

// --------------------------------------------------------------------------
// Static data: type chart, Pokemon and damage tables
// --------------------------------------------------------------------------

static u8 GetMonId(u8 battler)
{
    return (battler & BIT_SIDE) * PARTY_SIZE + gBattlerPartyIndexes[battler];
}

static bool8 IsBattlerPresent(u8 battler)
{
    return battler < gBattlersCount && !(gAbsentBattlerFlags & gBitTable[battler]) && gBattleMons[battler].hp != 0;
}

static void BuildTypeChart(void)
{
    s32 i, j;

    for (i = 0; i < NUMBER_OF_MON_TYPES; i++)
    {
        for (j = 0; j < NUMBER_OF_MON_TYPES; j++)
            sTypeChart[i][j] = TYPE_MUL_NORMAL;
    }
    for (i = 0; TYPE_EFFECT_ATK_TYPE(i) != TYPE_ENDTABLE; i += 3)
    {
        if (TYPE_EFFECT_ATK_TYPE(i) == TYPE_FORESIGHT)
            continue;
        sTypeChart[TYPE_EFFECT_ATK_TYPE(i)][TYPE_EFFECT_DEF_TYPE(i)] = TYPE_EFFECT_MULTIPLIER(i);
    }
}

static s32 GetTypeEffect(u8 type, const struct EvalMon *def)
{
    s32 eff = Q_ONE * sTypeChart[type][def->types[0]] / TYPE_MUL_NORMAL;

    if (def->types[1] != def->types[0])
        eff = eff * sTypeChart[type][def->types[1]] / TYPE_MUL_NORMAL;
    return eff;
}

static bool8 IsOfType(const struct EvalMon *mon, u8 type)
{
    return mon->types[0] == type || mon->types[1] == type;
}

static void SetHoldEffect(struct EvalMon *m, u16 item)
{
    m->holdEffect = ItemId_GetHoldEffect(item);
    m->holdEffectParam = ItemId_GetHoldEffectParam(item);
}

static void BuildMons(void)
{
    s32 side, i, j;

    for (side = 0; side < NUM_BATTLE_SIDES; side++)
    {
        struct Pokemon *party = (side == B_SIDE_PLAYER) ? gPlayerParty : gEnemyParty;

        for (i = 0; i < PARTY_SIZE; i++)
        {
            struct EvalMon *m = &sMons[side * PARTY_SIZE + i];
            u16 species = GetMonData(&party[i], MON_DATA_SPECIES_OR_EGG);

            m->present = FALSE;
            if (species == SPECIES_NONE || species == SPECIES_EGG || IsMonBenched(&party[i]))
                continue;
            m->present = TRUE;
            m->species = species;
            m->level = GetMonData(&party[i], MON_DATA_LEVEL);
            m->maxHP = GetMonData(&party[i], MON_DATA_MAX_HP);
            m->stats[STAT_HP] = m->maxHP;
            m->stats[STAT_ATK] = GetMonData(&party[i], MON_DATA_ATK);
            m->stats[STAT_DEF] = GetMonData(&party[i], MON_DATA_DEF);
            m->stats[STAT_SPEED] = GetMonData(&party[i], MON_DATA_SPEED);
            m->stats[STAT_SPATK] = GetMonData(&party[i], MON_DATA_SPATK);
            m->stats[STAT_SPDEF] = GetMonData(&party[i], MON_DATA_SPDEF);
            for (j = 0; j < MAX_MON_MOVES; j++)
                m->moves[j] = GetMonData(&party[i], MON_DATA_MOVE1 + j);
            m->types[0] = gSpeciesInfo[species].types[0];
            m->types[1] = gSpeciesInfo[species].types[1];
            m->ability = GetAbilityBySpecies(species, GetMonData(&party[i], MON_DATA_ABILITY_NUM));
            SetHoldEffect(m, GetMonData(&party[i], MON_DATA_HELD_ITEM));
            m->friendship = GetMonData(&party[i], MON_DATA_FRIENDSHIP);
        }
    }

    // Pokemon in battle use their battle data, which reflects Transform, Trace, Knock Off and so on
    for (i = 0; i < gBattlersCount; i++)
    {
        struct EvalMon *m;

        if (!IsBattlerPresent(i))
            continue;
        m = &sMons[GetMonId(i)];
        m->present = TRUE;
        m->species = gBattleMons[i].species;
        m->level = gBattleMons[i].level;
        m->maxHP = gBattleMons[i].maxHP;
        m->stats[STAT_ATK] = gBattleMons[i].attack;
        m->stats[STAT_DEF] = gBattleMons[i].defense;
        m->stats[STAT_SPEED] = gBattleMons[i].speed;
        m->stats[STAT_SPATK] = gBattleMons[i].spAttack;
        m->stats[STAT_SPDEF] = gBattleMons[i].spDefense;
        for (j = 0; j < MAX_MON_MOVES; j++)
            m->moves[j] = gBattleMons[i].moves[j];
        m->types[0] = gBattleMons[i].type1;
        m->types[1] = gBattleMons[i].type2;
        m->ability = gBattleMons[i].ability;
        SetHoldEffect(m, gBattleMons[i].item);
    }
}

static u8 StatusFromStatus1(u32 status1, u8 *turns)
{
    *turns = 0;
    if (status1 & STATUS1_SLEEP)
    {
        *turns = status1 & STATUS1_SLEEP;
        return EST_SLEEP;
    }
    if (status1 & STATUS1_TOXIC_POISON)
    {
        *turns = (status1 & STATUS1_TOXIC_COUNTER) >> 8;
        return EST_TOXIC;
    }
    if (status1 & STATUS1_POISON)
        return EST_POISON;
    if (status1 & STATUS1_BURN)
        return EST_BURN;
    if (status1 & STATUS1_FREEZE)
        return EST_FREEZE;
    if (status1 & STATUS1_PARALYSIS)
        return EST_PARALYSIS;
    return EST_NONE;
}

static void BuildRootState(void)
{
    s32 i, j;

    for (i = 0; i < EVAL_MONS; i++)
    {
        struct EvalMonState *st = &sRoot.mons[i];
        struct Pokemon *mon = (i < PARTY_SIZE) ? &gPlayerParty[i] : &gEnemyParty[i - PARTY_SIZE];
        u16 hp;

        st->flags = 0;
        st->confusion = 0;
        st->subHP = 0;
        for (j = 0; j < NUM_BATTLE_STATS; j++)
            st->stages[j] = 0;
        hp = sMons[i].present ? GetMonData(mon, MON_DATA_HP) : 0;
        st->hp = hp * 16;
        st->alive = hp != 0 ? Q_ONE : 0;
        st->status = StatusFromStatus1(GetMonData(mon, MON_DATA_STATUS), &st->statusTurns);
        st->statusProb = st->status != EST_NONE ? Q_ONE : 0;
    }

    for (i = 0; i < MAX_BATTLERS_COUNT; i++)
    {
        struct EvalMonState *st;
        u8 id;

        sRoot.active[i] = NO_MON;
        sRoot.wishTurns[i] = 0;
        sRoot.wishHP[i] = 0;
        if (!IsBattlerPresent(i))
            continue;
        id = GetMonId(i);
        sRoot.active[i] = id;
        st = &sRoot.mons[id];
        st->hp = gBattleMons[i].hp * 16;
        st->alive = Q_ONE;
        st->status = StatusFromStatus1(gBattleMons[i].status1, &st->statusTurns);
        st->statusProb = st->status != EST_NONE ? Q_ONE : 0;
        for (j = 0; j < NUM_BATTLE_STATS; j++)
            st->stages[j] = (gBattleMons[i].statStages[j] - DEFAULT_STAT_STAGE) * STAGE_UNIT;
        if (gBattleMons[i].status2 & STATUS2_CONFUSION)
            st->confusion = Q_ONE;
        if (gBattleMons[i].status2 & STATUS2_SUBSTITUTE)
        {
            st->flags |= EMF_SUBSTITUTE;
            st->subHP = gDisableStructs[i].substituteHP;
        }
        if (gBattleMons[i].status2 & STATUS2_CURSED)
            st->flags |= EMF_CURSED;
        if (gBattleMons[i].status2 & STATUS2_FOCUS_ENERGY)
            st->flags |= EMF_FOCUS;
        if (gStatuses3[i] & STATUS3_LEECHSEED)
            st->flags |= EMF_SEEDED;
        if (gDisableStructs[i].tauntTimer != 0)
            st->flags |= EMF_TAUNTED;
        if (gDisableStructs[i].isFirstTurn)
            st->flags |= EMF_FIRST_TURN;
        if (gBattleMons[i].ability == ABILITY_TRUANT && gDisableStructs[i].truantCounter)
            st->flags |= EMF_LOAFING;
        sWork->protectChance[i] = Q_ONE;
        if (gDisableStructs[i].protectUses != 0
         && (gLastResultingMoves[i] == MOVE_PROTECT || gLastResultingMoves[i] == MOVE_DETECT || gLastResultingMoves[i] == MOVE_ENDURE))
            sWork->protectChance[i] = Q_ONE >> (gDisableStructs[i].protectUses > 3 ? 3 : gDisableStructs[i].protectUses);
        if (gWishFutureKnock.wishCounter[i] != 0)
        {
            sRoot.wishTurns[i] = gWishFutureKnock.wishCounter[i];
            sRoot.wishHP[i] = sMons[id].maxHP * 8;
        }
    }

    for (i = 0; i < NUM_BATTLE_SIDES; i++)
    {
        sRoot.reflect[i] = (gSideStatuses[i] & SIDE_STATUS_REFLECT) ? gSideTimers[i].reflectTimer : 0;
        sRoot.lightScreen[i] = (gSideStatuses[i] & SIDE_STATUS_LIGHTSCREEN) ? gSideTimers[i].lightscreenTimer : 0;
        sRoot.safeguard[i] = (gSideStatuses[i] & SIDE_STATUS_SAFEGUARD) ? gSideTimers[i].safeguardTimer : 0;
        sRoot.spikes[i] = (gSideStatuses[i] & SIDE_STATUS_SPIKES) ? gSideTimers[i].spikesAmount : 0;
        sRoot.sleepClause[i] = IsSleepClauseActive(i); // battler i is on side i
    }

    sRoot.weather = EW_NONE;
    if (!ABILITY_ON_FIELD(ABILITY_CLOUD_NINE) && !ABILITY_ON_FIELD(ABILITY_AIR_LOCK))
    {
        if (gBattleWeather & B_WEATHER_RAIN)
            sRoot.weather = EW_RAIN;
        else if (gBattleWeather & B_WEATHER_SUN)
            sRoot.weather = EW_SUN;
        else if (gBattleWeather & B_WEATHER_SANDSTORM)
            sRoot.weather = EW_SAND;
        else if (gBattleWeather & B_WEATHER_HAIL)
            sRoot.weather = EW_HAIL;
    }
}

static u16 GetFlailPower(u16 hp16, u16 maxHP)
{
    s32 ratio = hp16 * 3 / maxHP; // hp * 48 / maxHP

    if (ratio <= 1)
        return 200;
    if (ratio <= 4)
        return 150;
    if (ratio <= 9)
        return 100;
    if (ratio <= 16)
        return 80;
    if (ratio <= 32)
        return 40;
    return 20;
}

static u16 GetLowKickPower(u16 species)
{
    u16 weight = GetPokedexHeightWeight(SpeciesToNationalPokedexNum(species), 1);

    if (weight < 100)
        return 20;
    if (weight < 250)
        return 40;
    if (weight < 500)
        return 60;
    if (weight < 1000)
        return 80;
    if (weight < 2000)
        return 100;
    return 120;
}

// Max-roll damage at neutral stat stages, before weather, screens and critical hits
static u16 CalcMoveDamage(u8 atkId, u8 slot, u8 defId)
{
    const struct EvalMon *atk = &sMons[atkId];
    const struct EvalMon *def = &sMons[defId];
    u16 move = atk->moves[slot];
    const struct BattleMove *bm = &gBattleMoves[move];
    s32 power = bm->power;
    u8 type = bm->type;
    s32 attack, defense, damage, eff;
    u8 boostedType;

    if (move == MOVE_NONE || power == 0)
        return 0;

    switch (bm->effect)
    {
    case EFFECT_LEVEL_DAMAGE:
    case EFFECT_PSYWAVE:
    case EFFECT_SONICBOOM:
    case EFFECT_DRAGON_RAGE:
        if (GetTypeEffect(type, def) == 0)
            return 0;
        sFixedDamage[atkId] |= 1 << slot;
        if (bm->effect == EFFECT_SONICBOOM)
            return 20;
        if (bm->effect == EFFECT_DRAGON_RAGE)
            return 40;
        return atk->level;
    case EFFECT_SUPER_FANG:
    case EFFECT_ENDEAVOR:
    case EFFECT_COUNTER:
    case EFFECT_MIRROR_COAT:
    case EFFECT_BIDE:
    case EFFECT_OHKO:
    case EFFECT_SPIT_UP:
        return 0; // worked out during the simulated turn
    case EFFECT_HIDDEN_POWER:
        type = TYPE_DARK; // every Pokemon has perfect IVs
        power = 70;
        break;
    case EFFECT_RETURN:
        power = atk->friendship * 10 / 25;
        break;
    case EFFECT_FRUSTRATION:
        power = (MAX_FRIENDSHIP - atk->friendship) * 10 / 25;
        break;
    case EFFECT_LOW_KICK:
        power = GetLowKickPower(def->species);
        break;
    case EFFECT_FLAIL:
        power = GetFlailPower(sRoot.mons[atkId].hp, atk->maxHP);
        break;
    case EFFECT_MAGNITUDE:
        power = 71;
        break;
    case EFFECT_PRESENT:
        power = 52;
        break;
    case EFFECT_TRIPLE_KICK:
        power = 60;
        break;
    case EFFECT_MULTI_HIT:
        power *= 3;
        break;
    case EFFECT_DOUBLE_HIT:
    case EFFECT_TWINEEDLE:
        power *= 2;
        break;
    case EFFECT_BEAT_UP:
        power = 40;
        break;
    }
    if (power <= 1)
        return 0;

    eff = GetTypeEffect(type, def);
    if (eff == 0)
        return 0;
    switch (def->ability)
    {
    case ABILITY_LEVITATE:
        if (type == TYPE_GROUND)
            return 0;
        break;
    case ABILITY_WONDER_GUARD:
        if (eff <= Q_ONE)
            return 0;
        break;
    case ABILITY_VOLT_ABSORB:
        if (type == TYPE_ELECTRIC)
            return 0;
        break;
    case ABILITY_WATER_ABSORB:
        if (type == TYPE_WATER)
            return 0;
        break;
    case ABILITY_FLASH_FIRE:
        if (type == TYPE_FIRE)
            return 0;
        break;
    }

    if (IS_MOVE_PHYSICAL(move))
    {
        attack = atk->stats[STAT_ATK];
        defense = def->stats[STAT_DEF];
        if (atk->ability == ABILITY_HUGE_POWER || atk->ability == ABILITY_PURE_POWER)
            attack *= 2;
        if (atk->ability == ABILITY_HUSTLE)
            attack = attack * 3 / 2;
        if (atk->ability == ABILITY_GUTS && sRoot.mons[atkId].status != EST_NONE)
            attack = attack * 3 / 2;
        if (atk->holdEffect == HOLD_EFFECT_CHOICE_BAND)
            attack = attack * 3 / 2;
        if (atk->holdEffect == HOLD_EFFECT_THICK_CLUB && (atk->species == SPECIES_CUBONE || atk->species == SPECIES_MAROWAK))
            attack *= 2;
        if (def->holdEffect == HOLD_EFFECT_METAL_POWDER && def->species == SPECIES_DITTO)
            defense *= 2;
        if (def->ability == ABILITY_MARVEL_SCALE && sRoot.mons[defId].status != EST_NONE)
            defense = defense * 3 / 2;
    }
    else
    {
        attack = atk->stats[STAT_SPATK];
        defense = def->stats[STAT_SPDEF];
        if (atk->holdEffect == HOLD_EFFECT_LIGHT_BALL && atk->species == SPECIES_PIKACHU)
            attack *= 2;
        if (atk->holdEffect == HOLD_EFFECT_DEEP_SEA_TOOTH && atk->species == SPECIES_CLAMPERL)
            attack *= 2;
        if (def->holdEffect == HOLD_EFFECT_DEEP_SEA_SCALE && def->species == SPECIES_CLAMPERL)
            defense *= 2;
    }
    boostedType = GetHoldEffectBoostedType(atk->holdEffect);
    if (boostedType == type)
        attack = attack * (100 + atk->holdEffectParam) / 100;
    if (def->ability == ABILITY_THICK_FAT && (type == TYPE_FIRE || type == TYPE_ICE))
        attack /= 2;
    if (sRoot.mons[atkId].hp * 3 <= atk->maxHP * 16)
    {
        if ((type == TYPE_GRASS && atk->ability == ABILITY_OVERGROW)
         || (type == TYPE_FIRE && atk->ability == ABILITY_BLAZE)
         || (type == TYPE_WATER && atk->ability == ABILITY_TORRENT)
         || (type == TYPE_BUG && atk->ability == ABILITY_SWARM))
            power = power * 3 / 2;
    }
    if (defense == 0)
        defense = 1;

    damage = (atk->level * 2 / 5 + 2) * power * attack / defense / 50 + 2;
    if (IsOfType(atk, type))
        damage = damage * 3 / 2;
    damage = (damage * eff) >> 12;
    if (damage < 1)
        damage = 1;
    if (damage > 0xFFFF)
        damage = 0xFFFF;
    return damage;
}

static void BuildDamageTables(void)
{
    s32 atk, slot, def;

    for (atk = 0; atk < EVAL_MONS; atk++)
    {
        if (sMons[atk].present)
            sMons[atk].hpRecip = (Q_ONE << 16) / (sMons[atk].maxHP * 16);
        sFixedDamage[atk] = 0;
        for (slot = 0; slot < MAX_MON_MOVES; slot++)
        {
            for (def = 0; def < EVAL_MONS; def++)
            {
                if (atk == def || !sMons[atk].present || !sMons[def].present)
                    sDamage[atk][slot][def] = 0;
                else
                    sDamage[atk][slot][def] = CalcMoveDamage(atk, slot, def);
            }
        }
    }

    // Each Pokemon's strongest attack against each foe, counting accuracy
    for (atk = 0; atk < EVAL_MONS; atk++)
    {
        for (def = 0; def < EVAL_MONS; def++)
        {
            s32 best = 0;

            sMons[atk].bestSlot[def] = MAX_MON_MOVES;
            for (slot = 0; slot < MAX_MON_MOVES; slot++)
            {
                u16 move = sMons[atk].moves[slot];
                s32 dmg = sDamage[atk][slot][def];

                if (dmg == 0 || gBattleMoves[move].effect == EFFECT_FAKE_OUT || gBattleMoves[move].effect == EFFECT_FOCUS_PUNCH)
                    continue;
                if (gBattleMoves[move].accuracy != 0)
                    dmg = dmg * gBattleMoves[move].accuracy / 100;
                if (dmg > best)
                {
                    best = dmg;
                    sMons[atk].bestSlot[def] = slot;
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Simulation helpers
// --------------------------------------------------------------------------

static s32 PercentToQ(s32 percent)
{
    return (percent * 10486) >> 8;
}

static u32 GetStageMult(const u16 *table, s32 stage)
{
    s32 i, f;

    if (stage <= -MAX_STAGE)
        return table[0];
    if (stage >= MAX_STAGE)
        return table[12];
    stage += MAX_STAGE;
    i = stage >> 4;
    f = stage & 15;
    return (table[i] * (16 - f) + table[i + 1] * f) >> 4;
}

static void ChangeStage(struct EvalMonState *st, u8 stat, s32 delta)
{
    s32 stage = st->stages[stat] + delta;

    if (stage > MAX_STAGE)
        stage = MAX_STAGE;
    if (stage < -MAX_STAGE)
        stage = -MAX_STAGE;
    st->stages[stat] = stage;
}

static bool8 BlocksStatDrops(const struct EvalMon *m, u8 stat)
{
    if (m->ability == ABILITY_CLEAR_BODY || m->ability == ABILITY_WHITE_SMOKE)
        return TRUE;
    if (m->ability == ABILITY_HYPER_CUTTER && stat == STAT_ATK)
        return TRUE;
    if (m->ability == ABILITY_KEEN_EYE && stat == STAT_ACC)
        return TRUE;
    return FALSE;
}

static void LowerStat(struct EvalState *s, u8 defId, u8 stat, s32 stages, u16 chance)
{
    if (chance == 0 || BlocksStatDrops(&sMons[defId], stat))
        return;
    ChangeStage(&s->mons[defId], stat, -QM(stages * STAGE_UNIT, chance));
}

static void RaiseStat(struct EvalState *s, u8 id, u8 stat, s32 stages, u16 chance)
{
    ChangeStage(&s->mons[id], stat, QM(stages * STAGE_UNIT, chance));
}

static void Residual(struct EvalMonState *st, s32 dmg16)
{
    st->hp -= dmg16;
    if (st->hp <= 0)
    {
        st->hp = 0;
        st->alive = 0;
    }
}

static void Heal(struct EvalMonState *st, u8 id, s32 amount16)
{
    s32 max = sMons[id].maxHP * 16;
    s32 hp = st->hp + amount16;

    st->hp = hp > max ? max : hp;
}

static u32 GetSpeed(const struct EvalState *s, u8 id)
{
    const struct EvalMon *m = &sMons[id];
    const struct EvalMonState *st = &s->mons[id];
    u32 speed = m->stats[STAT_SPEED];

    if (st->stages[STAT_SPEED] != 0)
        speed = (speed * GetStageMult(sStageMult, st->stages[STAT_SPEED])) >> 12;

    if ((m->ability == ABILITY_SWIFT_SWIM && s->weather == EW_RAIN)
     || (m->ability == ABILITY_CHLOROPHYLL && s->weather == EW_SUN))
        speed *= 2;
    if (m->holdEffect == HOLD_EFFECT_MACHO_BRACE)
        speed /= 2;
    if (st->status == EST_PARALYSIS)
        speed -= QM(speed * 3 / 4, st->statusProb);
    return speed;
}

static u16 GetHitChance(const struct EvalState *s, u8 atkId, u8 defId, u16 move)
{
    const struct BattleMove *bm = &gBattleMoves[move];
    s32 acc = bm->accuracy;
    s32 chance, stage;

    if (acc == 0 || bm->effect == EFFECT_ALWAYS_HIT || bm->effect == EFFECT_VITAL_THROW)
        return Q_ONE;
    if (bm->effect == EFFECT_THUNDER)
    {
        if (s->weather == EW_RAIN)
            return Q_ONE;
        if (s->weather == EW_SUN)
            acc = 50;
    }
    chance = (acc * 10486) >> 8; // acc * Q_ONE / 100
    stage = s->mons[atkId].stages[STAT_ACC] - s->mons[defId].stages[STAT_EVASION];
    if (stage != 0)
        chance = (chance * GetStageMult(sAccStageMult, stage)) >> 12;
    if (sMons[atkId].ability == ABILITY_COMPOUND_EYES)
        chance = QM(chance, 5325);
    if (sMons[atkId].ability == ABILITY_HUSTLE && IS_MOVE_PHYSICAL(move))
        chance = QM(chance, 3277);
    if (chance > Q_ONE)
        chance = Q_ONE;
    return chance;
}

// Probability (given it is still standing) that the Pokemon can act this turn
static u16 GetActChance(struct EvalState *s, u8 battler, u8 id)
{
    struct EvalMonState *st = &s->mons[id];
    s32 p = Q_ONE;

    if (st->flags & EMF_LOAFING)
        return 0;
    switch (st->status)
    {
    case EST_SLEEP:
        if (st->statusTurns > ((sMons[id].ability == ABILITY_EARLY_BIRD) ? 2 : 1))
            p = Q_ONE - st->statusProb;
        break;
    case EST_FREEZE:
        p = Q_ONE - QM(st->statusProb, 3277);
        break;
    case EST_PARALYSIS:
        p = Q_ONE - QM(st->statusProb, 1024);
        break;
    }
    p = QM(p, Q_ONE - st->confusion / 2);
    p = QM(p, Q_ONE - sTurn.flinch[battler]);
    return p;
}

static void TickSleep(struct EvalMonState *st, u8 id)
{
    u8 tick = (sMons[id].ability == ABILITY_EARLY_BIRD) ? 2 : 1;

    if (st->status != EST_SLEEP)
        return;
    if (st->statusTurns > tick)
    {
        st->statusTurns -= tick;
    }
    else
    {
        st->statusTurns = 0;
        st->status = EST_NONE;
        st->statusProb = 0;
    }
}

static void TryStatus(struct EvalState *s, u8 defId, u8 status, u16 chance)
{
    const struct EvalMon *m = &sMons[defId];
    struct EvalMonState *st = &s->mons[defId];
    u8 side = defId >= PARTY_SIZE ? B_SIDE_OPPONENT : B_SIDE_PLAYER;
    u16 p;

    if (chance == 0 || st->alive == 0 || s->safeguard[side])
        return;
    switch (status)
    {
    case EST_SLEEP:
        if (m->ability == ABILITY_INSOMNIA || m->ability == ABILITY_VITAL_SPIRIT || s->sleepClause[side])
            return;
        break;
    case EST_POISON:
    case EST_TOXIC:
        if (IsOfType(m, TYPE_POISON) || IsOfType(m, TYPE_STEEL) || m->ability == ABILITY_IMMUNITY)
            return;
        break;
    case EST_BURN:
        if (IsOfType(m, TYPE_FIRE) || m->ability == ABILITY_WATER_VEIL)
            return;
        break;
    case EST_FREEZE:
        if (IsOfType(m, TYPE_ICE) || m->ability == ABILITY_MAGMA_ARMOR || s->weather == EW_SUN)
            return;
        break;
    case EST_PARALYSIS:
        if (m->ability == ABILITY_LIMBER)
            return;
        break;
    }
    if ((m->holdEffect == HOLD_EFFECT_CURE_STATUS || (m->holdEffect == HOLD_EFFECT_CURE_SLP && status == EST_SLEEP))
     && !(st->flags & EMF_NO_ITEM))
    {
        if (chance >= Q_HALF)
            st->flags |= EMF_NO_ITEM;
        return;
    }

    if (st->status == EST_NONE || st->statusProb == 0)
    {
        st->status = status;
        st->statusProb = chance;
        st->statusTurns = (status == EST_SLEEP) ? 3 : (status == EST_TOXIC) ? 1 : 0;
    }
    else if (st->status == status)
    {
        st->statusProb += QM(Q_ONE - st->statusProb, chance);
    }
    else
    {
        p = QM(chance, Q_ONE - st->statusProb);
        if (p > st->statusProb)
        {
            st->status = status;
            st->statusProb = p;
            st->statusTurns = (status == EST_SLEEP) ? 3 : (status == EST_TOXIC) ? 1 : 0;
        }
    }
    if (status == EST_SLEEP && st->status == EST_SLEEP && st->statusProb >= Q_HALF)
        s->sleepClause[side] = TRUE;
}

static void AddConfusion(struct EvalState *s, u8 defId, u16 chance)
{
    struct EvalMonState *st = &s->mons[defId];

    if (sMons[defId].ability == ABILITY_OWN_TEMPO || s->safeguard[defId >= PARTY_SIZE])
        return;
    st->confusion += QM(Q_ONE - st->confusion, chance);
}

static u16 GetCritChance(u8 atkId, u8 defId, u16 move, const struct EvalState *s)
{
    s32 stage = 0;
    u8 effect = gBattleMoves[move].effect;

    if (sMons[defId].ability == ABILITY_BATTLE_ARMOR || sMons[defId].ability == ABILITY_SHELL_ARMOR)
        return 0;
    if (effect == EFFECT_HIGH_CRITICAL || effect == EFFECT_BLAZE_KICK || effect == EFFECT_POISON_TAIL || effect == EFFECT_SKY_ATTACK)
        stage++;
    if (sMons[atkId].holdEffect == HOLD_EFFECT_SCOPE_LENS)
        stage++;
    if (s->mons[atkId].flags & EMF_FOCUS)
        stage += 2;
    if (stage > 4)
        stage = 4;
    return sCritChance[stage];
}

static bool8 AreBothSlotsFilled(const struct EvalState *s, u8 side)
{
    u8 a = s->active[side], b = s->active[side | BIT_FLANK];

    return sIsDouble && a != NO_MON && b != NO_MON && s->mons[a].alive >= Q_HALF && s->mons[b].alive >= Q_HALF;
}

// Max-roll damage in HP against the target now, with the field and stat stages
static s32 GetSimDamage(const struct EvalState *s, u8 atkBattler, u8 atkId, u8 slot, u8 defBattler, u8 defId)
{
    u16 move = sMons[atkId].moves[slot];
    const struct BattleMove *bm = &gBattleMoves[move];
    const struct EvalMonState *a = &s->mons[atkId];
    const struct EvalMonState *d = &s->mons[defId];
    u8 defSide = defBattler & BIT_SIDE;
    u8 type = (bm->effect == EFFECT_HIDDEN_POWER) ? TYPE_DARK : bm->type;
    s32 dmg = sDamage[atkId][slot][defId];

    if (dmg == 0)
        return 0;
    if (sFixedDamage[atkId] & (1 << slot))
        return dmg;

    if (IS_MOVE_PHYSICAL(move))
    {
        if (a->stages[STAT_ATK] != 0)
            dmg = (dmg * GetStageMult(sStageMult, a->stages[STAT_ATK])) >> 12;
        if (d->stages[STAT_DEF] != 0)
            dmg = Div(dmg << 12, GetStageMult(sStageMult, d->stages[STAT_DEF]));
        if (a->status == EST_BURN && sMons[atkId].ability != ABILITY_GUTS)
            dmg -= QM(dmg / 2, a->statusProb);
        if (s->reflect[defSide])
            dmg = AreBothSlotsFilled(s, defSide) ? QM(dmg, 2731) : dmg / 2;
    }
    else
    {
        if (a->stages[STAT_SPATK] != 0)
            dmg = (dmg * GetStageMult(sStageMult, a->stages[STAT_SPATK])) >> 12;
        if (d->stages[STAT_SPDEF] != 0)
            dmg = Div(dmg << 12, GetStageMult(sStageMult, d->stages[STAT_SPDEF]));
        if (s->lightScreen[defSide])
            dmg = AreBothSlotsFilled(s, defSide) ? QM(dmg, 2731) : dmg / 2;
    }

    if (s->weather == EW_RAIN)
    {
        if (type == TYPE_WATER)
            dmg = dmg * 3 / 2;
        else if (type == TYPE_FIRE)
            dmg /= 2;
    }
    else if (s->weather == EW_SUN)
    {
        if (type == TYPE_FIRE)
            dmg = dmg * 3 / 2;
        else if (type == TYPE_WATER)
            dmg /= 2;
    }

    switch (bm->effect)
    {
    case EFFECT_ERUPTION:
        dmg = QM(dmg, (a->hp * sMons[atkId].hpRecip) >> 16);
        break;
    case EFFECT_FACADE:
        if (a->status == EST_BURN || a->status == EST_POISON || a->status == EST_TOXIC || a->status == EST_PARALYSIS)
            dmg += QM(dmg, a->statusProb);
        break;
    case EFFECT_SMELLINGSALT:
        if (d->status == EST_PARALYSIS)
            dmg += QM(dmg, d->statusProb);
        break;
    case EFFECT_REVENGE:
        dmg += QM(dmg, sTurn.hit[atkBattler]);
        break;
    case EFFECT_WEATHER_BALL:
        if (s->weather != EW_NONE)
            dmg *= 2;
        break;
    case EFFECT_SOLAR_BEAM:
        if (s->weather != EW_SUN)
            dmg /= 2;
        break;
    case EFFECT_RAZOR_WIND:
    case EFFECT_SKULL_BASH:
    case EFFECT_SKY_ATTACK:
    case EFFECT_SEMI_INVULNERABLE:
        dmg /= 2;
        break;
    case EFFECT_RECHARGE:
        dmg = QM(dmg, 2458);
        break;
    case EFFECT_FUTURE_SIGHT:
        dmg = QM(dmg, 2867);
        break;
    case EFFECT_DREAM_EATER:
        dmg = (d->status == EST_SLEEP) ? QM(dmg, d->statusProb) : 0;
        break;
    case EFFECT_SNORE:
        dmg = (a->status == EST_SLEEP) ? dmg : 0;
        break;
    }

    if (sTurn.helping[atkBattler])
        dmg += QM(dmg / 2, sTurn.helping[atkBattler]);
    if (sTurn.powerBoost[atkBattler])
        dmg += QM(dmg, sTurn.powerBoost[atkBattler]);
    if (bm->target == MOVE_TARGET_BOTH && AreBothSlotsFilled(s, defSide))
        dmg /= 2;
    return dmg;
}

static u16 GetKOChance(s32 dmg16, s32 hp16)
{
    s32 range;

    if (hp16 <= 0)
        return Q_ONE;
    if (dmg16 < hp16)
        return 0;
    range = QM(dmg16, 614); // the damage roll spans 85-100%
    if (dmg16 - range >= hp16 || range <= 0)
        return Q_ONE;
    return Div((dmg16 - hp16) * Q_ONE, range);
}

// Applies a hit that lands with probability `p` (if the target is still standing).
// Returns the expected damage dealt in 1/16 HP; `survive` is the chance the target
// is still standing after a hit, for secondary effects.
static s32 ApplyHit(struct EvalState *s, u8 defBattler, u8 defId, s32 dmg, u16 p, u16 crit, bool8 fixed, u16 *survive)
{
    struct EvalMonState *t = &s->mons[defId];
    s32 dmg16 = dmg * 16;
    s32 h = t->hp;
    s32 pKO, pk, surv, remain, dealt;

    *survive = 0;
    if (p == 0 || t->alive == 0 || dmg <= 0)
        return 0;

    if (t->flags & EMF_SUBSTITUTE)
    {
        if (p >= Q_HALF)
        {
            if (dmg >= t->subHP)
            {
                t->flags &= ~EMF_SUBSTITUTE;
                t->subHP = 0;
            }
            else
            {
                t->subHP -= dmg;
            }
        }
        return 0;
    }

    if (fixed)
    {
        pKO = dmg16 >= h ? Q_ONE : 0;
        surv = dmg16;
    }
    else
    {
        pKO = QM(Q_ONE - crit, GetKOChance(dmg16, h)) + QM(crit, GetKOChance(dmg16 * 2, h));
        if (pKO > 0)
            surv = (QM(dmg16, 3482) + (dmg16 < h ? dmg16 : h)) / 2;
        else
            surv = QM(QM(dmg16, 3789), Q_ONE + crit);
    }
    pKO = QM(pKO, Q_ONE - sTurn.endure[defBattler]);
    if (surv > h - 16)
        surv = h - 16;
    if (surv < 0)
        surv = 0;

    pk = QM(p, pKO);
    remain = Q_ONE - pk;
    dealt = QM(t->alive, QM(pk, h) + QM(p - pk, surv));
    if (remain <= 0)
    {
        t->alive = 0;
        t->hp = 0;
        return dealt;
    }
    t->hp = Div((QM(Q_ONE - p, h) + QM(p - pk, h - surv)) * Q_ONE, remain);
    t->alive = QM(t->alive, remain);
    *survive = Q_ONE - pKO;
    return dealt;
}

// --------------------------------------------------------------------------
// Moves
// --------------------------------------------------------------------------

static bool8 IsSingleTarget(u8 target)
{
    return target == MOVE_TARGET_SELECTED || target == MOVE_TARGET_DEPENDS
        || target == MOVE_TARGET_USER_OR_SELECTED || target == MOVE_TARGET_RANDOM;
}

static u8 GetFoe(u8 battler)
{
    return battler ^ BIT_SIDE;
}

// Who a move hits and with what weight. Returns the number of targets.
static u8 GetTargets(const struct EvalState *s, u8 battler, u16 move, u8 chosen, u8 *targets, u16 *weights)
{
    u8 moveTarget = gBattleMoves[move].target;
    u8 foe = GetFoe(battler);
    u8 n = 0;
    u8 other;

    if (moveTarget == MOVE_TARGET_BOTH || moveTarget == MOVE_TARGET_FOES_AND_ALLY)
    {
        if (s->active[foe] != NO_MON)
        {
            targets[n] = foe;
            weights[n++] = Q_ONE;
        }
        if (sIsDouble && s->active[foe ^ BIT_FLANK] != NO_MON)
        {
            targets[n] = foe ^ BIT_FLANK;
            weights[n++] = Q_ONE;
        }
        if (moveTarget == MOVE_TARGET_FOES_AND_ALLY && sIsDouble && s->active[battler ^ BIT_FLANK] != NO_MON)
        {
            targets[n] = battler ^ BIT_FLANK;
            weights[n++] = Q_ONE;
        }
        return n;
    }
    if (!IsSingleTarget(moveTarget))
        return 0;

    if (!sIsDouble)
    {
        if (s->active[foe] == NO_MON)
            return 0;
        targets[0] = foe;
        weights[0] = Q_ONE;
        return 1;
    }

    if (chosen != NO_TARGET && (chosen & BIT_SIDE) == (battler & BIT_SIDE))
    {
        if (chosen == battler || s->active[chosen] == NO_MON)
            return 0;
        targets[0] = chosen;
        weights[0] = Q_ONE;
        return 1;
    }

    if (chosen == NO_TARGET || moveTarget == MOVE_TARGET_RANDOM)
        chosen = foe;
    other = chosen ^ BIT_FLANK;
    if (moveTarget == MOVE_TARGET_RANDOM && s->active[chosen] != NO_MON && s->active[other] != NO_MON)
    {
        targets[0] = chosen;
        weights[0] = Q_HALF;
        targets[1] = other;
        weights[1] = Q_HALF;
        return 2;
    }

    // Follow Me draws single-target moves
    if (sTurn.followMe[chosen & BIT_SIDE] != 0xFF && sTurn.followMe[chosen & BIT_SIDE] != chosen)
    {
        u16 fm = sTurn.followMeChance[chosen & BIT_SIDE];

        targets[n] = sTurn.followMe[chosen & BIT_SIDE];
        weights[n++] = fm;
        if (s->active[chosen] != NO_MON)
        {
            targets[n] = chosen;
            weights[n++] = Q_ONE - fm;
        }
        return n;
    }

    if (s->active[chosen] == NO_MON)
    {
        if (s->active[other] == NO_MON)
            return 0;
        targets[0] = other;
        weights[0] = Q_ONE;
        return 1;
    }
    targets[n] = chosen;
    weights[n++] = Q_ONE;
    // If the target faints first, the move goes to its partner instead
    if (s->active[other] != NO_MON && s->mons[s->active[chosen]].alive < Q_ONE)
    {
        targets[n] = other;
        weights[n++] = Q_ONE - s->mons[s->active[chosen]].alive;
    }
    return n;
}

static void ApplySecondary(struct EvalState *s, u8 atkBattler, u8 atkId, u8 defBattler, u8 defId, u16 move, u16 chance)
{
    const struct BattleMove *bm = &gBattleMoves[move];
    u16 sp;

    if (chance == 0 || sMons[defId].ability == ABILITY_SHIELD_DUST)
        return;
    sp = bm->secondaryEffectChance ? QM(chance, PercentToQ(bm->secondaryEffectChance)) : chance;
    if (sMons[atkId].ability == ABILITY_SERENE_GRACE)
        sp = (sp * 2 > chance) ? chance : sp * 2;

    switch (bm->effect)
    {
    case EFFECT_POISON_HIT:
    case EFFECT_POISON_TAIL:
        TryStatus(s, defId, EST_POISON, sp);
        break;
    case EFFECT_POISON_FANG:
        TryStatus(s, defId, EST_TOXIC, sp);
        break;
    case EFFECT_BURN_HIT:
    case EFFECT_BLAZE_KICK:
    case EFFECT_THAW_HIT:
        TryStatus(s, defId, EST_BURN, sp);
        break;
    case EFFECT_FREEZE_HIT:
        TryStatus(s, defId, EST_FREEZE, sp);
        break;
    case EFFECT_PARALYZE_HIT:
    case EFFECT_SECRET_POWER:
        TryStatus(s, defId, EST_PARALYSIS, sp);
        break;
    case EFFECT_TRI_ATTACK:
        TryStatus(s, defId, EST_PARALYSIS, sp / 3);
        TryStatus(s, defId, EST_BURN, sp / 3);
        TryStatus(s, defId, EST_FREEZE, sp / 3);
        break;
    case EFFECT_CONFUSE_HIT:
        AddConfusion(s, defId, sp);
        break;
    case EFFECT_FLINCH_HIT:
    case EFFECT_FLINCH_MINIMIZE_HIT:
    case EFFECT_TWISTER:
    case EFFECT_FAKE_OUT:
        if (!sTurn.done[defBattler] && sMons[defId].ability != ABILITY_INNER_FOCUS)
            sTurn.flinch[defBattler] += QM(Q_ONE - sTurn.flinch[defBattler], sp);
        break;
    case EFFECT_ATTACK_DOWN_HIT:
        LowerStat(s, defId, STAT_ATK, 1, sp);
        break;
    case EFFECT_DEFENSE_DOWN_HIT:
        LowerStat(s, defId, STAT_DEF, 1, sp);
        break;
    case EFFECT_SPEED_DOWN_HIT:
        LowerStat(s, defId, STAT_SPEED, 1, sp);
        break;
    case EFFECT_SPECIAL_ATTACK_DOWN_HIT:
        LowerStat(s, defId, STAT_SPATK, 1, sp);
        break;
    case EFFECT_SPECIAL_DEFENSE_DOWN_HIT:
        LowerStat(s, defId, STAT_SPDEF, 1, sp);
        break;
    case EFFECT_ACCURACY_DOWN_HIT:
        LowerStat(s, defId, STAT_ACC, 1, sp);
        break;
    case EFFECT_KNOCK_OFF:
        if (sp >= Q_HALF && sMons[defId].ability != ABILITY_STICKY_HOLD)
            s->mons[defId].flags |= EMF_NO_ITEM;
        break;
    }
}

static void DoDamagingMove(struct EvalState *s, u8 battler, u8 id, u8 slot, u8 chosenTarget, u16 weight)
{
    struct EvalMonState *st = &s->mons[id];
    const struct EvalMon *m = &sMons[id];
    u16 move = m->moves[slot];
    const struct BattleMove *bm = &gBattleMoves[move];
    u8 targets[3];
    u16 weights[3];
    u8 n, k;
    s32 totalDealt = 0;
    u16 landed = 0;

    // Counter and Mirror Coat hit back whoever hit the user this turn
    if (bm->effect == EFFECT_COUNTER || bm->effect == EFFECT_MIRROR_COAT)
    {
        s32 taken = (bm->effect == EFFECT_COUNTER) ? sTurn.physTaken[battler] : sTurn.specTaken[battler];
        u8 from = (bm->effect == EFFECT_COUNTER) ? sTurn.physFrom[battler] : sTurn.specFrom[battler];
        u16 survive;

        if (taken > 0 && from != NO_TARGET && s->active[from] != NO_MON)
            ApplyHit(s, from, s->active[from], taken * 2 / 16, weight, 0, TRUE, &survive);
        return;
    }

    n = GetTargets(s, battler, move, chosenTarget, targets, weights);
    for (k = 0; k < n; k++)
    {
        u8 tb = targets[k];
        u8 defId = s->active[tb];
        struct EvalMonState *d;
        s32 dmg, dealt;
        u16 p, crit = 0, survive;
        bool8 fixed = FALSE;

        if (defId == NO_MON)
            continue;
        d = &s->mons[defId];
        p = QM(weight, weights[k]);
        if (bm->flags & FLAG_PROTECT_AFFECTED)
            p = QM(p, Q_ONE - sTurn.protect[tb]);
        p = QM(p, GetHitChance(s, id, defId, move));
        if (p == 0)
            continue;

        switch (bm->effect)
        {
        case EFFECT_SUPER_FANG:
            dmg = GetTypeEffect(bm->type, &sMons[defId]) ? d->hp / 32 : 0;
            fixed = TRUE;
            break;
        case EFFECT_ENDEAVOR:
            dmg = (d->hp > st->hp && GetTypeEffect(bm->type, &sMons[defId])) ? (d->hp - st->hp) / 16 : 0;
            fixed = TRUE;
            break;
        default:
            dmg = GetSimDamage(s, battler, id, slot, tb, defId);
            fixed = (sFixedDamage[id] & (1 << slot)) != 0;
            if (!fixed)
                crit = GetCritChance(id, defId, move, s);
            break;
        }
        if (dmg <= 0)
            continue;

        dealt = ApplyHit(s, tb, defId, dmg, p, crit, fixed, &survive);
        sTurn.hit[tb] += QM(Q_ONE - sTurn.hit[tb], p);
        if (IS_MOVE_PHYSICAL(move))
        {
            sTurn.physTaken[tb] += dealt;
            sTurn.physFrom[tb] = battler;
        }
        else
        {
            sTurn.specTaken[tb] += dealt;
            sTurn.specFrom[tb] = battler;
        }
        totalDealt += dealt;
        if (k == 0)
            landed = p;
        ApplySecondary(s, battler, id, tb, defId, move, QM(p, survive));

        // Contact abilities
        if ((bm->flags & FLAG_MAKES_CONTACT) && survive)
        {
            switch (sMons[defId].ability)
            {
            case ABILITY_ROUGH_SKIN:
                Residual(st, QM(m->maxHP, p));
                break;
            case ABILITY_STATIC:
                TryStatus(s, id, EST_PARALYSIS, QM(p, 1229));
                break;
            case ABILITY_FLAME_BODY:
                TryStatus(s, id, EST_BURN, QM(p, 1229));
                break;
            case ABILITY_POISON_POINT:
                TryStatus(s, id, EST_POISON, QM(p, 1229));
                break;
            case ABILITY_EFFECT_SPORE:
                TryStatus(s, id, EST_SLEEP, QM(p, 410));
                break;
            }
        }
        if (bm->effect == EFFECT_BRICK_BREAK && p >= Q_HALF)
        {
            s->reflect[tb & BIT_SIDE] = 0;
            s->lightScreen[tb & BIT_SIDE] = 0;
        }
    }

    // Effects on the user
    switch (bm->effect)
    {
    case EFFECT_ABSORB:
    case EFFECT_DREAM_EATER:
        Heal(st, id, totalDealt / 2);
        break;
    case EFFECT_RECOIL:
        if (m->ability != ABILITY_ROCK_HEAD)
            Residual(st, totalDealt / 4);
        break;
    case EFFECT_DOUBLE_EDGE:
        if (m->ability != ABILITY_ROCK_HEAD)
            Residual(st, QM(totalDealt, 1365));
        break;
    case EFFECT_SUPERPOWER:
        ChangeStage(st, STAT_ATK, -QM(STAGE_UNIT, landed));
        ChangeStage(st, STAT_DEF, -QM(STAGE_UNIT, landed));
        break;
    case EFFECT_OVERHEAT:
        ChangeStage(st, STAT_SPATK, -QM(2 * STAGE_UNIT, landed));
        break;
    case EFFECT_ATTACK_UP_HIT:
        RaiseStat(s, id, STAT_ATK, 1, QM(landed, PercentToQ(bm->secondaryEffectChance)));
        break;
    case EFFECT_DEFENSE_UP_HIT:
        RaiseStat(s, id, STAT_DEF, 1, QM(landed, PercentToQ(bm->secondaryEffectChance)));
        break;
    case EFFECT_ALL_STATS_UP_HIT:
    {
        u16 c = QM(landed, PercentToQ(bm->secondaryEffectChance));

        RaiseStat(s, id, STAT_ATK, 1, c);
        RaiseStat(s, id, STAT_DEF, 1, c);
        RaiseStat(s, id, STAT_SPEED, 1, c);
        RaiseStat(s, id, STAT_SPATK, 1, c);
        RaiseStat(s, id, STAT_SPDEF, 1, c);
        break;
    }
    case EFFECT_RAPID_SPIN:
        if (landed >= Q_HALF)
        {
            s->spikes[battler & BIT_SIDE] = 0;
            st->flags &= ~EMF_SEEDED;
        }
        break;
    case EFFECT_RAMPAGE:
        st->confusion += QM(Q_ONE - st->confusion, weight / 4);
        break;
    }
    if (m->holdEffect == HOLD_EFFECT_SHELL_BELL && !(st->flags & EMF_NO_ITEM))
        Heal(st, id, totalDealt / 8);
}

static void DoStatusMove(struct EvalState *s, u8 battler, u8 id, u8 slot, u8 chosenTarget, u16 weight)
{
    struct EvalMonState *st = &s->mons[id];
    const struct EvalMon *m = &sMons[id];
    u16 move = m->moves[slot];
    const struct BattleMove *bm = &gBattleMoves[move];
    u8 side = battler & BIT_SIDE;
    s32 maxHP16 = m->maxHP * 16;
    u8 targets[3];
    u16 weights[3];
    u8 n, k;

    // Moves that affect the user or the field
    switch (bm->effect)
    {
    case EFFECT_ATTACK_UP:
        RaiseStat(s, id, STAT_ATK, 1, weight);
        return;
    case EFFECT_ATTACK_UP_2:
        RaiseStat(s, id, STAT_ATK, 2, weight);
        return;
    case EFFECT_DEFENSE_UP:
    case EFFECT_DEFENSE_CURL:
        RaiseStat(s, id, STAT_DEF, 1, weight);
        return;
    case EFFECT_DEFENSE_UP_2:
        RaiseStat(s, id, STAT_DEF, 2, weight);
        return;
    case EFFECT_SPEED_UP:
        RaiseStat(s, id, STAT_SPEED, 1, weight);
        return;
    case EFFECT_SPEED_UP_2:
        RaiseStat(s, id, STAT_SPEED, 2, weight);
        return;
    case EFFECT_SPECIAL_ATTACK_UP:
        RaiseStat(s, id, STAT_SPATK, 1, weight);
        return;
    case EFFECT_SPECIAL_ATTACK_UP_2:
        RaiseStat(s, id, STAT_SPATK, 2, weight);
        return;
    case EFFECT_SPECIAL_DEFENSE_UP:
        RaiseStat(s, id, STAT_SPDEF, 1, weight);
        return;
    case EFFECT_SPECIAL_DEFENSE_UP_2:
        RaiseStat(s, id, STAT_SPDEF, 2, weight);
        return;
    case EFFECT_ACCURACY_UP:
    case EFFECT_ACCURACY_UP_2:
        RaiseStat(s, id, STAT_ACC, 1, weight);
        return;
    case EFFECT_CALM_MIND:
        RaiseStat(s, id, STAT_SPATK, 1, weight);
        RaiseStat(s, id, STAT_SPDEF, 1, weight);
        return;
    case EFFECT_BULK_UP:
        RaiseStat(s, id, STAT_ATK, 1, weight);
        RaiseStat(s, id, STAT_DEF, 1, weight);
        return;
    case EFFECT_DRAGON_DANCE:
        RaiseStat(s, id, STAT_ATK, 1, weight);
        RaiseStat(s, id, STAT_SPEED, 1, weight);
        return;
    case EFFECT_COSMIC_POWER:
        RaiseStat(s, id, STAT_DEF, 1, weight);
        RaiseStat(s, id, STAT_SPDEF, 1, weight);
        return;
    case EFFECT_CURSE:
        if (IsOfType(m, TYPE_GHOST))
        {
            u8 foe = GetFoe(battler);

            if (chosenTarget != NO_TARGET && (chosenTarget & BIT_SIDE) != side)
                foe = chosenTarget;
            if (s->active[foe] == NO_MON)
                return;
            Residual(st, QM(maxHP16 / 2, weight));
            if (weight >= Q_HALF)
                s->mons[s->active[foe]].flags |= EMF_CURSED;
            return;
        }
        RaiseStat(s, id, STAT_ATK, 1, weight);
        RaiseStat(s, id, STAT_DEF, 1, weight);
        ChangeStage(st, STAT_SPEED, -QM(STAGE_UNIT, weight));
        return;
    case EFFECT_BELLY_DRUM:
        if (st->hp > maxHP16 / 2)
        {
            Residual(st, QM(maxHP16 / 2, weight));
            ChangeStage(st, STAT_ATK, QM(MAX_STAGE - st->stages[STAT_ATK], weight));
        }
        return;
    case EFFECT_FOCUS_ENERGY:
        if (weight >= Q_HALF)
            st->flags |= EMF_FOCUS;
        return;
    case EFFECT_HAZE:
        for (k = 0; k < MAX_BATTLERS_COUNT; k++)
        {
            u8 j;

            if (s->active[k] == NO_MON)
                continue;
            for (j = 0; j < NUM_BATTLE_STATS; j++)
                s->mons[s->active[k]].stages[j] -= QM(s->mons[s->active[k]].stages[j], weight);
        }
        return;
    case EFFECT_RESTORE_HP:
    case EFFECT_SOFTBOILED:
        Heal(st, id, QM(maxHP16 / 2, weight));
        return;
    case EFFECT_MORNING_SUN:
    case EFFECT_SYNTHESIS:
    case EFFECT_MOONLIGHT:
        if (s->weather == EW_SUN)
            Heal(st, id, QM(maxHP16 * 2 / 3, weight));
        else if (s->weather == EW_NONE)
            Heal(st, id, QM(maxHP16 / 2, weight));
        else
            Heal(st, id, QM(maxHP16 / 4, weight));
        return;
    case EFFECT_REST:
        if (st->hp < maxHP16 && m->ability != ABILITY_INSOMNIA && m->ability != ABILITY_VITAL_SPIRIT)
        {
            Heal(st, id, QM(maxHP16 - st->hp, weight));
            if (st->status != EST_SLEEP)
                st->statusProb = QM(st->statusProb, Q_ONE - weight) + weight;
            st->status = EST_SLEEP;
            st->statusTurns = 3;
        }
        return;
    case EFFECT_WISH:
        if (s->wishTurns[battler] == 0)
        {
            s->wishTurns[battler] = 2;
            s->wishHP[battler] = QM(maxHP16 / 2, weight);
        }
        return;
    case EFFECT_PROTECT:
    case EFFECT_ENDURE:
    {
        u16 success = Q_ONE;

        // Consecutive uses only matter for the real turn being graded
        if (sRoot.active[battler] == id)
            success = sWork->protectChance[battler];
        if (bm->effect == EFFECT_PROTECT)
            sTurn.protect[battler] = QM(weight, success);
        else
            sTurn.endure[battler] = QM(weight, success);
        return;
    }
    case EFFECT_SUBSTITUTE:
        if (!(st->flags & EMF_SUBSTITUTE) && st->hp > maxHP16 / 4)
        {
            Residual(st, QM(maxHP16 / 4, weight));
            if (weight >= Q_HALF)
            {
                st->flags |= EMF_SUBSTITUTE;
                st->subHP = m->maxHP / 4;
            }
        }
        return;
    case EFFECT_REFLECT:
        if (weight >= Q_HALF && s->reflect[side] == 0)
            s->reflect[side] = 5;
        return;
    case EFFECT_LIGHT_SCREEN:
        if (weight >= Q_HALF && s->lightScreen[side] == 0)
            s->lightScreen[side] = 5;
        return;
    case EFFECT_SAFEGUARD:
        if (weight >= Q_HALF && s->safeguard[side] == 0)
            s->safeguard[side] = 5;
        return;
    case EFFECT_SPIKES:
        if (weight >= Q_HALF && s->spikes[side ^ BIT_SIDE] < 3)
            s->spikes[side ^ BIT_SIDE]++;
        return;
    case EFFECT_RAIN_DANCE:
        if (weight >= Q_HALF)
            s->weather = EW_RAIN;
        return;
    case EFFECT_SUNNY_DAY:
        if (weight >= Q_HALF)
            s->weather = EW_SUN;
        return;
    case EFFECT_SANDSTORM:
        if (weight >= Q_HALF)
            s->weather = EW_SAND;
        return;
    case EFFECT_HAIL:
        if (weight >= Q_HALF)
            s->weather = EW_HAIL;
        return;
    case EFFECT_HEAL_BELL:
        for (k = 0; k < PARTY_SIZE; k++)
        {
            struct EvalMonState *ally = &s->mons[side * PARTY_SIZE + k];

            ally->statusProb = QM(ally->statusProb, Q_ONE - weight);
        }
        return;
    case EFFECT_REFRESH:
        if (st->status == EST_BURN || st->status == EST_PARALYSIS || st->status == EST_POISON || st->status == EST_TOXIC)
            st->statusProb = QM(st->statusProb, Q_ONE - weight);
        return;
    case EFFECT_HELPING_HAND:
        if (sIsDouble && s->active[battler ^ BIT_FLANK] != NO_MON && !sTurn.done[battler ^ BIT_FLANK])
            sTurn.helping[battler ^ BIT_FLANK] = weight;
        return;
    case EFFECT_FOLLOW_ME:
        if (sIsDouble)
        {
            sTurn.followMe[side] = battler;
            sTurn.followMeChance[side] = weight;
        }
        return;
    case EFFECT_MEMENTO:
        break;
    case EFFECT_SLEEP:
    case EFFECT_YAWN:
    case EFFECT_POISON:
    case EFFECT_TOXIC:
    case EFFECT_PARALYZE:
    case EFFECT_WILL_O_WISP:
    case EFFECT_CONFUSE:
    case EFFECT_FLATTER:
    case EFFECT_SWAGGER:
    case EFFECT_ATTACK_DOWN:
    case EFFECT_DEFENSE_DOWN:
    case EFFECT_SPEED_DOWN:
    case EFFECT_SPECIAL_ATTACK_DOWN:
    case EFFECT_SPECIAL_DEFENSE_DOWN:
    case EFFECT_ACCURACY_DOWN:
    case EFFECT_EVASION_DOWN:
    case EFFECT_ATTACK_DOWN_2:
    case EFFECT_DEFENSE_DOWN_2:
    case EFFECT_SPEED_DOWN_2:
    case EFFECT_SPECIAL_ATTACK_DOWN_2:
    case EFFECT_SPECIAL_DEFENSE_DOWN_2:
    case EFFECT_ACCURACY_DOWN_2:
    case EFFECT_EVASION_DOWN_2:
    case EFFECT_TICKLE:
    case EFFECT_LEECH_SEED:
    case EFFECT_TAUNT:
    case EFFECT_ROAR:
    case EFFECT_PAIN_SPLIT:
    case EFFECT_PSYCH_UP:
    case EFFECT_TEETER_DANCE:
        break;
    default:
        return;
    }

    // Moves that affect a target
    n = GetTargets(s, battler, move, chosenTarget, targets, weights);
    if (bm->effect == EFFECT_TEETER_DANCE)
    {
        n = 0;
        for (k = 0; k < MAX_BATTLERS_COUNT; k++)
        {
            if (k != battler && s->active[k] != NO_MON)
            {
                targets[n] = k;
                weights[n++] = Q_ONE;
            }
        }
    }
    for (k = 0; k < n; k++)
    {
        u8 tb = targets[k];
        u8 defId = s->active[tb];
        struct EvalMonState *d;
        u16 p;
        bool8 bySub;

        if (defId == NO_MON)
            continue;
        d = &s->mons[defId];
        p = QM(QM(weight, weights[k]), Q_ONE - sTurn.protect[tb]);
        p = QM(p, GetHitChance(s, id, defId, move));
        bySub = (d->flags & EMF_SUBSTITUTE) && bm->effect != EFFECT_ROAR && bm->effect != EFFECT_PSYCH_UP;
        if (p == 0 || bySub)
            continue;

        switch (bm->effect)
        {
        case EFFECT_SLEEP:
            TryStatus(s, defId, EST_SLEEP, p);
            break;
        case EFFECT_YAWN:
            TryStatus(s, defId, EST_SLEEP, QM(p, 2458));
            break;
        case EFFECT_POISON:
            TryStatus(s, defId, EST_POISON, p);
            break;
        case EFFECT_TOXIC:
            TryStatus(s, defId, EST_TOXIC, p);
            break;
        case EFFECT_PARALYZE:
            if (GetTypeEffect(bm->type, &sMons[defId]) != 0)
                TryStatus(s, defId, EST_PARALYSIS, p);
            break;
        case EFFECT_WILL_O_WISP:
            TryStatus(s, defId, EST_BURN, p);
            break;
        case EFFECT_CONFUSE:
        case EFFECT_TEETER_DANCE:
            AddConfusion(s, defId, p);
            break;
        case EFFECT_FLATTER:
            ChangeStage(d, STAT_SPATK, QM(STAGE_UNIT, p));
            AddConfusion(s, defId, p);
            break;
        case EFFECT_SWAGGER:
            ChangeStage(d, STAT_ATK, QM(2 * STAGE_UNIT, p));
            AddConfusion(s, defId, p);
            break;
        case EFFECT_ATTACK_DOWN:
            LowerStat(s, defId, STAT_ATK, 1, p);
            break;
        case EFFECT_DEFENSE_DOWN:
            LowerStat(s, defId, STAT_DEF, 1, p);
            break;
        case EFFECT_SPEED_DOWN:
            LowerStat(s, defId, STAT_SPEED, 1, p);
            break;
        case EFFECT_SPECIAL_ATTACK_DOWN:
            LowerStat(s, defId, STAT_SPATK, 1, p);
            break;
        case EFFECT_SPECIAL_DEFENSE_DOWN:
            LowerStat(s, defId, STAT_SPDEF, 1, p);
            break;
        case EFFECT_ACCURACY_DOWN:
            LowerStat(s, defId, STAT_ACC, 1, p);
            break;
        case EFFECT_ATTACK_DOWN_2:
            LowerStat(s, defId, STAT_ATK, 2, p);
            break;
        case EFFECT_DEFENSE_DOWN_2:
            LowerStat(s, defId, STAT_DEF, 2, p);
            break;
        case EFFECT_SPEED_DOWN_2:
            LowerStat(s, defId, STAT_SPEED, 2, p);
            break;
        case EFFECT_SPECIAL_ATTACK_DOWN_2:
            LowerStat(s, defId, STAT_SPATK, 2, p);
            break;
        case EFFECT_SPECIAL_DEFENSE_DOWN_2:
            LowerStat(s, defId, STAT_SPDEF, 2, p);
            break;
        case EFFECT_ACCURACY_DOWN_2:
            LowerStat(s, defId, STAT_ACC, 2, p);
            break;
        case EFFECT_TICKLE:
            LowerStat(s, defId, STAT_ATK, 1, p);
            LowerStat(s, defId, STAT_DEF, 1, p);
            break;
        case EFFECT_MEMENTO:
            LowerStat(s, defId, STAT_ATK, 2, p);
            LowerStat(s, defId, STAT_SPATK, 2, p);
            break;
        case EFFECT_LEECH_SEED:
            if (!IsOfType(&sMons[defId], TYPE_GRASS) && p >= Q_HALF)
                d->flags |= EMF_SEEDED;
            break;
        case EFFECT_TAUNT:
            if (p >= Q_HALF)
                d->flags |= EMF_TAUNTED;
            break;
        case EFFECT_ROAR:
        {
            u8 j;

            if (sMons[defId].ability == ABILITY_SUCTION_CUPS)
                break;
            for (j = 0; j < NUM_BATTLE_STATS; j++)
                d->stages[j] -= QM(d->stages[j], p);
            break;
        }
        case EFFECT_PAIN_SPLIT:
        {
            s32 avg = (st->hp + d->hp) / 2;

            st->hp += QM(avg - st->hp, p);
            if (st->hp > maxHP16)
                st->hp = maxHP16;
            d->hp += QM(avg - d->hp, p);
            if (d->hp > sMons[defId].maxHP * 16)
                d->hp = sMons[defId].maxHP * 16;
            break;
        }
        case EFFECT_PSYCH_UP:
        {
            u8 j;

            for (j = 0; j < NUM_BATTLE_STATS; j++)
                st->stages[j] += QM(d->stages[j] - st->stages[j], weight);
            break;
        }
        }
    }
    if (bm->effect == EFFECT_MEMENTO)
        st->alive = QM(st->alive, Q_ONE - weight);
}

static void ExecuteMove(struct EvalState *s, u8 battler, const struct EvalAction *act)
{
    u8 id = s->active[battler];
    struct EvalMonState *st;
    u16 move, weight;
    u8 slot;
    const struct BattleMove *bm;

    if (id == NO_MON || act->kind != ACT_MOVE)
        return;
    st = &s->mons[id];
    if (st->alive == 0)
        return;
    slot = act->index;
    move = sMons[id].moves[slot];
    bm = &gBattleMoves[move];

    if (bm->effect == EFFECT_SLEEP_TALK || bm->effect == EFFECT_SNORE)
    {
        // Only works while asleep
        if (st->status != EST_SLEEP)
            return;
        weight = QM(QM(st->alive, st->statusProb), Q_ONE - sTurn.flinch[battler]);
        if (bm->effect == EFFECT_SLEEP_TALK)
        {
            // Sleep Talk calls a random other move: count it as the best attack at 3/4 strength
            u8 i, best = MAX_MON_MOVES;
            s32 bestDmg = 0;

            for (i = 0; i < MAX_MON_MOVES; i++)
            {
                u8 foe = GetFoe(battler);
                s32 dmg;

                if (i == slot || s->active[foe] == NO_MON)
                    continue;
                dmg = sDamage[id][i][s->active[foe]];
                if (dmg > bestDmg)
                {
                    bestDmg = dmg;
                    best = i;
                }
            }
            TickSleep(st, id);
            if (best != MAX_MON_MOVES)
                DoDamagingMove(s, battler, id, best, act->target, weight * 3 / 4);
            return;
        }
    }
    else
    {
        weight = QM(st->alive, GetActChance(s, battler, id));
    }
    TickSleep(st, id);
    if (weight == 0)
        return;
    if ((st->flags & EMF_TAUNTED) && bm->power == 0)
        return;
    if (bm->effect == EFFECT_FOCUS_PUNCH)
        weight = QM(weight, Q_ONE - sTurn.hit[battler]);
    if (bm->effect == EFFECT_FAKE_OUT && !(st->flags & EMF_FIRST_TURN))
        return;

    if (bm->power != 0)
        DoDamagingMove(s, battler, id, slot, act->target, weight);
    else
        DoStatusMove(s, battler, id, slot, act->target, weight);
}

static void DoSwitch(struct EvalState *s, u8 battler, u8 newId)
{
    u8 oldId = s->active[battler];
    u8 side = battler & BIT_SIDE;
    const struct EvalMon *m = &sMons[newId];
    struct EvalMonState *st = &s->mons[newId];
    u8 i;

    if (oldId != NO_MON)
    {
        struct EvalMonState *old = &s->mons[oldId];

        for (i = 0; i < NUM_BATTLE_STATS; i++)
            old->stages[i] = 0;
        old->confusion = 0;
        old->flags &= EMF_NO_ITEM;
        old->subHP = 0;
        if (sMons[oldId].ability == ABILITY_NATURAL_CURE)
            old->statusProb = 0;
    }
    s->active[battler] = newId;
    st->flags |= EMF_FIRST_TURN;

    if (s->spikes[side] && !IsOfType(m, TYPE_FLYING) && m->ability != ABILITY_LEVITATE)
        Residual(st, QM(m->maxHP * 16, sSpikesDamage[s->spikes[side]]));

    switch (m->ability)
    {
    case ABILITY_INTIMIDATE:
        for (i = 0; i < MAX_BATTLERS_COUNT; i++)
        {
            if ((i & BIT_SIDE) != side && s->active[i] != NO_MON)
                LowerStat(s, s->active[i], STAT_ATK, 1, Q_ONE);
        }
        break;
    case ABILITY_DRIZZLE:
        s->weather = EW_RAIN;
        break;
    case ABILITY_DROUGHT:
        s->weather = EW_SUN;
        break;
    case ABILITY_SAND_STREAM:
        s->weather = EW_SAND;
        break;
    }
}

static void EndTurn(struct EvalState *s)
{
    u8 b;

    for (b = 0; b < gBattlersCount; b++)
    {
        u8 id = s->active[b];
        const struct EvalMon *m;
        struct EvalMonState *st;
        s32 maxHP16;

        if (id == NO_MON)
            continue;
        m = &sMons[id];
        st = &s->mons[id];
        if (s->wishTurns[b] != 0 && --s->wishTurns[b] == 0 && st->alive)
            Heal(st, id, s->wishHP[b]);
        if (st->alive == 0)
            continue;
        maxHP16 = m->maxHP * 16;

        if (s->weather == EW_SAND && !IsOfType(m, TYPE_ROCK) && !IsOfType(m, TYPE_GROUND) && !IsOfType(m, TYPE_STEEL)
         && m->ability != ABILITY_SAND_VEIL)
            Residual(st, maxHP16 / 16);
        else if (s->weather == EW_HAIL && !IsOfType(m, TYPE_ICE))
            Residual(st, maxHP16 / 16);
        if (st->alive == 0)
            continue;
        if (m->holdEffect == HOLD_EFFECT_LEFTOVERS && !(st->flags & EMF_NO_ITEM))
            Heal(st, id, maxHP16 / 16);
        if (m->ability == ABILITY_RAIN_DISH && s->weather == EW_RAIN)
            Heal(st, id, maxHP16 / 16);
        if (st->flags & EMF_SEEDED)
        {
            s32 drain = st->hp < maxHP16 / 8 ? st->hp : maxHP16 / 8;
            u8 seeder = s->active[GetFoe(b)];

            Residual(st, drain);
            if (seeder != NO_MON && s->mons[seeder].alive)
                Heal(&s->mons[seeder], seeder, drain);
        }
        switch (st->status)
        {
        case EST_BURN:
        case EST_POISON:
            Residual(st, QM(maxHP16 / 8, st->statusProb));
            break;
        case EST_TOXIC:
            Residual(st, QM(maxHP16 * st->statusTurns / 16, st->statusProb));
            if (st->statusTurns < 15)
                st->statusTurns++;
            break;
        }
        if (st->flags & EMF_CURSED)
            Residual(st, maxHP16 / 4);
        if (m->ability == ABILITY_SPEED_BOOST && !(st->flags & EMF_FIRST_TURN))
            ChangeStage(st, STAT_SPEED, STAGE_UNIT);
        if (m->ability == ABILITY_SHED_SKIN)
            st->statusProb = QM(st->statusProb, 2731);
        st->confusion = st->confusion * 3 / 4;
        if (m->ability == ABILITY_TRUANT)
            st->flags ^= EMF_LOAFING;
        st->flags &= ~EMF_FIRST_TURN;
    }

    for (b = 0; b < NUM_BATTLE_SIDES; b++)
    {
        if (s->reflect[b])
            s->reflect[b]--;
        if (s->lightScreen[b])
            s->lightScreen[b]--;
        if (s->safeguard[b])
            s->safeguard[b]--;
    }
}

static void SimulateTurn(struct EvalState *s, const struct EvalAction *acts)
{
    u8 order[MAX_BATTLERS_COUNT];
    s32 prio[MAX_BATTLERS_COUNT];
    u32 speed[MAX_BATTLERS_COUNT];
    u8 n = 0;
    u8 i, j, b, f;

    for (i = 0; i < MAX_BATTLERS_COUNT; i++)
    {
        sTurn.protect[i] = 0;
        sTurn.endure[i] = 0;
        sTurn.flinch[i] = 0;
        sTurn.hit[i] = 0;
        sTurn.helping[i] = 0;
        sTurn.powerBoost[i] = 0;
        sTurn.physTaken[i] = 0;
        sTurn.specTaken[i] = 0;
        sTurn.physFrom[i] = NO_TARGET;
        sTurn.specFrom[i] = NO_TARGET;
        sTurn.done[i] = FALSE;
    }
    sTurn.followMe[0] = sTurn.followMe[1] = 0xFF;

    // Switches go first, faster Pokemon first
    for (b = 0; b < gBattlersCount; b++)
    {
        if (acts[b].kind == ACT_SWITCH && s->active[b] != NO_MON)
        {
            order[n] = b;
            speed[n] = GetSpeed(s, s->active[b]);
            n++;
        }
    }
    for (i = 1; i < n; i++)
    {
        for (j = i; j > 0 && speed[j] > speed[j - 1]; j--)
        {
            u8 tb = order[j];
            u32 ts = speed[j];

            order[j] = order[j - 1];
            order[j - 1] = tb;
            speed[j] = speed[j - 1];
            speed[j - 1] = ts;
        }
    }
    for (i = 0; i < n; i++)
    {
        b = order[i];
        // Pursuit hits a Pokemon on its way out at double power
        for (f = 0; f < gBattlersCount; f++)
        {
            u8 fid = s->active[f];

            if ((f & BIT_SIDE) == (b & BIT_SIDE) || fid == NO_MON || acts[f].kind != ACT_MOVE || sTurn.done[f])
                continue;
            if (gBattleMoves[sMons[fid].moves[acts[f].index]].effect != EFFECT_PURSUIT)
                continue;
            if (sIsDouble && acts[f].target != b)
                continue;
            sTurn.powerBoost[f] = Q_ONE;
            {
                struct EvalAction pursuit = acts[f];

                pursuit.target = b;
                ExecuteMove(s, f, &pursuit);
            }
            sTurn.powerBoost[f] = 0;
            sTurn.done[f] = TRUE;
        }
        DoSwitch(s, b, acts[b].index);
        sTurn.done[b] = TRUE;
    }

    // Moves by priority, then speed
    n = 0;
    for (b = 0; b < gBattlersCount; b++)
    {
        u8 id = s->active[b];

        if (acts[b].kind != ACT_MOVE || sTurn.done[b] || id == NO_MON)
            continue;
        order[n] = b;
        prio[n] = gBattleMoves[sMons[id].moves[acts[b].index]].priority;
        speed[n] = GetSpeed(s, id);
        n++;
    }
    for (i = 1; i < n; i++)
    {
        for (j = i; j > 0 && (prio[j] > prio[j - 1] || (prio[j] == prio[j - 1] && speed[j] > speed[j - 1])); j--)
        {
            u8 tb = order[j];
            s32 tp = prio[j];
            u32 ts = speed[j];

            order[j] = order[j - 1];
            order[j - 1] = tb;
            prio[j] = prio[j - 1];
            prio[j - 1] = tp;
            speed[j] = speed[j - 1];
            speed[j - 1] = ts;
        }
    }
    for (i = 0; i < n; i++)
    {
        ExecuteMove(s, order[i], &acts[order[i]]);
        sTurn.done[order[i]] = TRUE;
    }

    EndTurn(s);
}

// --------------------------------------------------------------------------
// Evaluation
// --------------------------------------------------------------------------

static s32 GetMonValue(const struct EvalState *s, u8 id)
{
    const struct EvalMon *m = &sMons[id];
    const struct EvalMonState *st = &s->mons[id];
    s32 value, penalty = 0;

    if (!m->present || st->alive == 0)
        return 0;
    value = W_ALIVE + QM(W_HP, (st->hp * m->hpRecip) >> 16);
    switch (st->status)
    {
    case EST_SLEEP:
        penalty = 500 * (st->statusTurns > 3 ? 3 : st->statusTurns);
        break;
    case EST_FREEZE:
        penalty = 1600;
        break;
    case EST_PARALYSIS:
        penalty = 600;
        break;
    case EST_BURN:
        penalty = (m->stats[STAT_ATK] > m->stats[STAT_SPATK]) ? 1000 : 300;
        break;
    case EST_POISON:
        penalty = 300;
        break;
    case EST_TOXIC:
        penalty = 600;
        break;
    }
    value = QM(value, Q_ONE - QM(penalty, st->statusProb));
    return QM(value, st->alive);
}

static bool8 IsActive(const struct EvalState *s, u8 id)
{
    u8 b;

    for (b = 0; b < MAX_BATTLERS_COUNT; b++)
    {
        if (s->active[b] == id)
            return TRUE;
    }
    return FALSE;
}

// Material balance from the player's point of view, in Pokemon (Q12)
static s32 GetMaterial(const struct EvalState *s)
{
    s32 total = 0;
    u8 side, i;

    for (side = 0; side < NUM_BATTLE_SIDES; side++)
    {
        s32 sideTotal = 0;

        for (i = 0; i < PARTY_SIZE; i++)
        {
            u8 id = side * PARTY_SIZE + i;

            sideTotal += GetMonValue(s, id);
            // Spikes wear down the Pokemon still to come in
            if (s->spikes[side] && sMons[id].present && s->mons[id].alive && !IsActive(s, id)
             && !IsOfType(&sMons[id], TYPE_FLYING) && sMons[id].ability != ABILITY_LEVITATE)
                sideTotal -= QM(QM(W_HP, sSpikesDamage[s->spikes[side]]), Q_HALF);
        }
        total += (side == B_SIDE_PLAYER) ? sideTotal : -sideTotal;
    }
    return total;
}

static u16 GetWinProb(s32 v)
{
    s32 a = v < 0 ? -v : v;
    s32 i = a >> 8;
    s32 p;

    if (i >= 128)
        p = Q_ONE;
    else
        p = sLogistic[i] + (((sLogistic[i + 1] - sLogistic[i]) * (a & 0xFF)) >> 8);
    return v < 0 ? Q_ONE - p : p;
}

// The attack that does the most expected damage, as a share of the target's HP
static void GetGreedyAction(const struct EvalState *s, u8 battler, struct EvalAction *act)
{
    u8 id = s->active[battler];
    u8 slot, k;
    s32 best = 0;

    act->kind = ACT_NONE;
    act->index = 0;
    act->target = NO_TARGET;
    if (id == NO_MON || s->mons[id].alive == 0)
        return;

    for (slot = 0; slot < MAX_MON_MOVES; slot++)
    {
        u16 move = sMons[id].moves[slot];
        const struct BattleMove *bm = &gBattleMoves[move];
        s32 score = 0;
        u8 bestTarget = NO_TARGET;
        s32 bestTargetScore = 0;

        if (move == MOVE_NONE || bm->power == 0)
            continue;
        if (bm->effect == EFFECT_FAKE_OUT && !(s->mons[id].flags & EMF_FIRST_TURN))
            continue;
        for (k = 0; k < 2; k++)
        {
            u8 foe = GetFoe(battler) ^ (k ? BIT_FLANK : 0);
            u8 defId;
            s32 dmg16, hp, frac;

            if (k == 1 && !sIsDouble)
                break;
            defId = s->active[foe];
            if (defId == NO_MON || s->mons[defId].alive == 0)
                continue;
            dmg16 = GetSimDamage(s, battler, id, slot, foe, defId) * 16;
            hp = s->mons[defId].hp;
            if (hp <= 0)
                continue;
            frac = dmg16 >= hp ? Q_ONE + Q_HALF : dmg16 * (Q_ONE / 16) / (hp / 16 + 1);
            if (frac > Q_ONE + Q_HALF)
                frac = Q_ONE + Q_HALF;
            frac = QM(QM(frac, GetHitChance(s, id, defId, move)), s->mons[defId].alive);
            if (IsSingleTarget(bm->target))
            {
                if (frac > bestTargetScore)
                {
                    bestTargetScore = frac;
                    bestTarget = foe;
                }
            }
            else
            {
                score += frac;
            }
        }
        if (IsSingleTarget(bm->target))
            score = bestTargetScore;
        if (score > best)
        {
            best = score;
            act->kind = ACT_MOVE;
            act->index = slot;
            act->target = IsSingleTarget(bm->target) ? bestTarget : NO_TARGET;
        }
    }
}

// Rollouts have every Pokemon use its strongest attack on whichever foe it hurts most
static void GetRolloutAction(const struct EvalState *s, u8 battler, struct EvalAction *act)
{
    u8 id = s->active[battler];
    u8 k;
    s32 best = -1;

    act->kind = ACT_NONE;
    act->target = NO_TARGET;
    if (id == NO_MON || s->mons[id].alive == 0)
        return;
    for (k = 0; k < (sIsDouble ? 2 : 1); k++)
    {
        u8 foe = GetFoe(battler) ^ (k ? BIT_FLANK : 0);
        u8 defId = s->active[foe];
        u8 slot;
        s32 frac, dmg;

        if (defId == NO_MON || s->mons[defId].alive == 0)
            continue;
        slot = sMons[id].bestSlot[defId];
        if (slot == MAX_MON_MOVES)
            continue;
        dmg = GetSimDamage(s, battler, id, slot, foe, defId);
        if (dmg > sMons[defId].maxHP)
            dmg = sMons[defId].maxHP;
        frac = (dmg * sMons[defId].hpRecip) >> 12; // share of max HP
        if (dmg * 16 >= s->mons[defId].hp)
            frac += Q_HALF;
        frac = QM(frac, s->mons[defId].alive);
        if (frac > best)
        {
            best = frac;
            act->kind = ACT_MOVE;
            act->index = slot;
            act->target = IsSingleTarget(gBattleMoves[sMons[id].moves[slot]].target) ? foe : NO_TARGET;
        }
    }
}

// A simulation is a real turn followed by two rollout turns, scored after each.
// It runs one turn per call so that it fits in a frame's spare time.
static void BeginSimulation(void)
{
    sSim = sRoot;
    sWork->simStage = 1;
    sWork->simMaterial = 0;
}

// Returns TRUE when the simulation is finished, with the win probability in `result`
static bool8 ContinueSimulation(u16 *result)
{
    struct EvalAction rollout[MAX_BATTLERS_COUNT];
    u8 b;

    if (sWork->simStage == 1)
    {
        SimulateTurn(&sSim, sWork->simActs);
        sWork->simMaterial = GetMaterial(&sSim);
        sWork->simStage++;
        return FALSE;
    }
    for (b = 0; b < gBattlersCount; b++)
        GetRolloutAction(&sSim, b, &rollout[b]);
    SimulateTurn(&sSim, rollout);
    if (sWork->simStage == 2)
    {
        sWork->simMaterial += GetMaterial(&sSim);
        sWork->simStage++;
        return FALSE;
    }
    sWork->simStage = 0;
    *result = GetWinProb((sWork->simMaterial + 2 * GetMaterial(&sSim)) / 4);
    return TRUE;
}

// --------------------------------------------------------------------------
// Options and grading
// --------------------------------------------------------------------------

static bool8 CanBattlerSwitch(u8 battler)
{
    if (gBattleMons[battler].status2 & (STATUS2_WRAPPED | STATUS2_ESCAPE_PREVENTION))
        return FALSE;
    if (gStatuses3[battler] & STATUS3_ROOTED)
        return FALSE;
    if (ABILITY_ON_OPPOSING_FIELD(battler, ABILITY_SHADOW_TAG))
        return FALSE;
    if (ABILITY_ON_OPPOSING_FIELD(battler, ABILITY_ARENA_TRAP)
     && !IS_BATTLER_OF_TYPE(battler, TYPE_FLYING) && gBattleMons[battler].ability != ABILITY_LEVITATE)
        return FALSE;
    if (AbilityBattleEffects(ABILITYEFFECT_CHECK_FIELD_EXCEPT_BATTLER, battler, ABILITY_MAGNET_PULL, 0, 0)
     && IS_BATTLER_OF_TYPE(battler, TYPE_STEEL))
        return FALSE;
    return TRUE;
}

static u8 NormalizeTarget(u8 battler, u8 slot, u8 target)
{
    u16 move = gBattleMons[battler].moves[slot];

    if (!IsSingleTarget(gBattleMoves[move].target))
        return NO_TARGET;
    if (!sIsDouble || gBattleMoves[move].target == MOVE_TARGET_RANDOM || target >= gBattlersCount)
        return GetFoe(battler);
    return target;
}

static void AddOption(u8 battler, u8 kind, u8 index, u8 target)
{
    struct EvalAction *opt;

    if (sNumOptions[battler] >= MAX_OPTIONS)
        return;
    opt = &sOptions[battler][sNumOptions[battler]++];
    opt->kind = kind;
    opt->index = index;
    opt->target = target;
}

static void BuildOptions(u8 battler)
{
    u8 id = sRoot.active[battler];
    u8 side = battler & BIT_SIDE;
    u8 unusable, slot, i;

    sNumOptions[battler] = 0;
    sForced[battler] = TRUE;
    if (id == NO_MON)
    {
        AddOption(battler, ACT_NONE, 0, NO_TARGET);
        return;
    }
    if (gBattleMons[battler].status2 & (STATUS2_MULTIPLETURNS | STATUS2_RECHARGE))
    {
        for (slot = 0; slot < MAX_MON_MOVES; slot++)
        {
            if (gBattleMons[battler].moves[slot] == gLockedMoves[battler] && gLockedMoves[battler] != MOVE_NONE)
            {
                AddOption(battler, ACT_MOVE, slot, NormalizeTarget(battler, slot, gBattleStruct->moveTarget[battler]));
                return;
            }
        }
        AddOption(battler, ACT_NONE, 0, NO_TARGET);
        return;
    }

    i = gPotentialItemEffectBattler;
    unusable = CheckMoveLimitations(battler, 0, 0xFF);
    gPotentialItemEffectBattler = i;
    if (gDisableStructs[battler].encoredMove != MOVE_NONE)
        unusable |= ~gBitTable[gDisableStructs[battler].encoredMovePos];

    for (slot = 0; slot < MAX_MON_MOVES; slot++)
    {
        u16 move = gBattleMons[battler].moves[slot];
        u8 target;

        if (unusable & gBitTable[slot])
            continue;
        target = gBattleMoves[move].target;
        if (sIsDouble && IsSingleTarget(target) && target != MOVE_TARGET_RANDOM)
        {
            u8 foe = GetFoe(battler);

            if (sRoot.active[foe] != NO_MON)
                AddOption(battler, ACT_MOVE, slot, foe);
            if (sRoot.active[foe ^ BIT_FLANK] != NO_MON)
                AddOption(battler, ACT_MOVE, slot, foe ^ BIT_FLANK);
        }
        else
        {
            AddOption(battler, ACT_MOVE, slot, NormalizeTarget(battler, slot, NO_TARGET));
        }
    }
    if (sNumOptions[battler] == 0)
        AddOption(battler, ACT_NONE, 0, NO_TARGET); // Struggle

    if (CanBattlerSwitch(battler))
    {
        for (i = 0; i < PARTY_SIZE; i++)
        {
            u8 monId = side * PARTY_SIZE + i;

            if (sMons[monId].present && sRoot.mons[monId].alive && !IsActive(&sRoot, monId))
                AddOption(battler, ACT_SWITCH, monId, NO_TARGET);
        }
    }
    sForced[battler] = (sNumOptions[battler] < 2);
}

static void GetChosenAction(u8 battler, struct EvalAction *act)
{
    act->kind = ACT_NONE;
    act->index = 0;
    act->target = NO_TARGET;
    if (sRoot.active[battler] == NO_MON)
        return;
    switch (gChosenActionByBattler[battler])
    {
    case B_ACTION_USE_MOVE:
        act->kind = ACT_MOVE;
        act->index = gBattleStruct->chosenMovePositions[battler];
        act->target = NormalizeTarget(battler, act->index, gBattleStruct->moveTarget[battler]);
        break;
    case B_ACTION_SWITCH:
        act->kind = ACT_SWITCH;
        act->index = (battler & BIT_SIDE) * PARTY_SIZE + gBattleStruct->monToSwitchIntoId[battler];
        break;
    }
}

static s32 FindOption(u8 battler, const struct EvalAction *act)
{
    s32 i;

    for (i = 0; i < sNumOptions[battler]; i++)
    {
        const struct EvalAction *opt = &sOptions[battler][i];

        if (opt->kind == act->kind && opt->index == act->index && opt->target == act->target)
            return i;
    }
    return -1;
}

static bool8 IsPlayerBattler(u8 battler)
{
    return (battler & BIT_SIDE) == B_SIDE_PLAYER;
}

static s32 GetPruneKeep(u8 battler)
{
    if (!sIsDouble)
        return MAX_OPTIONS;
    return PRUNED_OPTIONS;
}

// Keeps each battler's most promising options, judged against the other
// battlers' greedy attacks, so that the joint options stay few enough to search.
static void SelectPrunedOptions(u8 battler, s32 keep)
{
    u8 taken[MAX_OPTIONS];
    s32 i, n = sNumOptions[battler];
    s32 chosen = IsPlayerBattler(battler) ? FindOption(battler, &sChosen[battler]) : -1;

    sNumPruned[battler] = 0;
    for (i = 0; i < n; i++)
        taken[i] = FALSE;
    if (chosen >= 0)
    {
        taken[chosen] = TRUE;
        sPruned[battler][sNumPruned[battler]++] = chosen;
    }
    while (sNumPruned[battler] < keep)
    {
        s32 best = -1;

        for (i = 0; i < n; i++)
        {
            if (!taken[i] && (best < 0 || sWork->values[battler][i] > sWork->values[battler][best]))
                best = i;
        }
        if (best < 0)
            break;
        taken[best] = TRUE;
        sPruned[battler][sNumPruned[battler]++] = best;
    }
}

// Builds the list of joint options for one side; returns how many there are
static u8 BuildJointOptions(u8 side, u8 (*joint)[2], u8 max)
{
    u8 b0 = side, b1 = side | BIT_FLANK;
    u8 n0 = sNumPruned[b0];
    u8 n1 = sIsDouble ? sNumPruned[b1] : 1;
    u8 i, j, n = 0;

    for (i = 0; i < n0; i++)
    {
        for (j = 0; j < n1 && n < max; j++)
        {
            if (sIsDouble)
            {
                const struct EvalAction *a = &sOptions[b0][sPruned[b0][i]];
                const struct EvalAction *c = &sOptions[b1][sPruned[b1][j]];

                // Both can't switch to the same Pokemon
                if (a->kind == ACT_SWITCH && c->kind == ACT_SWITCH && a->index == c->index)
                    continue;
            }
            joint[n][0] = sPruned[b0][i];
            joint[n][1] = sIsDouble ? sPruned[b1][j] : 0;
            n++;
        }
    }
    return n;
}

static void FillActions(struct EvalAction *acts, const u8 *playerJoint, const u8 *oppJoint)
{
    acts[0] = sOptions[0][playerJoint[0]];
    acts[1] = sOptions[1][oppJoint[0]];
    if (sIsDouble)
    {
        acts[2] = sOptions[2][playerJoint[1]];
        acts[3] = sOptions[3][oppJoint[1]];
    }
    else
    {
        acts[2].kind = acts[3].kind = ACT_NONE;
    }
}

static void NormalizeStrategy(const s32 *regret, s32 *strategy, u8 n)
{
    s32 total = 0;
    u8 i;

    for (i = 0; i < n; i++)
        total += regret[i];
    for (i = 0; i < n; i++)
        strategy[i] = (total > 0) ? Div(regret[i] * Q_ONE, total) : Q_ONE / n;
}

// Regret matching+ on the payoff matrix; the opponent's average strategy is kept
static void SolveEquilibrium(u8 np, u8 no)
{
    s32 regP[MAX_JOINT_PLAYER], regO[MAX_JOINT_OPPONENT];
    s32 stratP[MAX_JOINT_PLAYER], stratO[MAX_JOINT_OPPONENT];
    s32 uP[MAX_JOINT_PLAYER], uO[MAX_JOINT_OPPONENT];
    u32 sumO[MAX_JOINT_OPPONENT];
    u32 sumP[MAX_JOINT_PLAYER];
    u32 total = 0;
    s32 it, v;
    u8 p, o;

    for (p = 0; p < np; p++)
    {
        regP[p] = 0;
        sumP[p] = 0;
    }
    for (o = 0; o < no; o++)
    {
        regO[o] = 0;
        sumO[o] = 0;
    }
    for (it = 0; it < EQUILIBRIUM_ITERS; it++)
    {
        NormalizeStrategy(regP, stratP, np);
        NormalizeStrategy(regO, stratO, no);
        v = 0;
        for (p = 0; p < np; p++)
        {
            s32 u = 0;

            for (o = 0; o < no; o++)
                u += QM(stratO[o], sMatrix[p][o]);
            uP[p] = u;
            v += QM(stratP[p], u);
        }
        for (o = 0; o < no; o++)
        {
            s32 u = 0;

            for (p = 0; p < np; p++)
                u += QM(stratP[p], sMatrix[p][o]);
            uO[o] = u;
        }
        for (p = 0; p < np; p++)
        {
            regP[p] += uP[p] - v;
            if (regP[p] < 0)
                regP[p] = 0;
            sumP[p] += stratP[p] * (it + 1) / 16;
        }
        for (o = 0; o < no; o++)
        {
            regO[o] += v - uO[o];
            if (regO[o] < 0)
                regO[o] = 0;
            sumO[o] += stratO[o] * (it + 1) / 16;
        }
    }
    for (o = 0; o < no; o++)
        total += sumO[o];
    for (o = 0; o < no; o++)
        sOppStrategy[o] = total ? (sumO[o] * Q_ONE) / total : Q_ONE / no;
    total = 0;
    for (p = 0; p < np; p++)
        total += sumP[p];
    for (p = 0; p < np; p++)
        sPlayerStrategy[p] = total ? (sumP[p] * Q_ONE) / total : Q_ONE / np;
}

static u8 GetGrade(u16 lossTenths)
{
    u8 grade;

    for (grade = 0; grade < EVAL_GRADE_COUNT - 1; grade++)
    {
        if (lossTenths <= sGradeThresholds[grade])
            return grade;
    }
    return EVAL_GRADE_BLUNDER;
}

static u16 GetLossTenths(s32 loss)
{
    if (loss < 0)
        loss = 0;
    return loss * 1000 / Q_ONE;
}

// Accuracy (0-100) of a decision that gave up `loss` win probability
static u8 GetDecisionAccuracy(s32 loss)
{
    u16 lossTenths = GetLossTenths(loss);
    u8 pct = lossTenths / 10;

    if (pct >= 100)
        return 0;
    return (sDecisionAccuracy[pct] * (10 - lossTenths % 10) + sDecisionAccuracy[pct + 1] * (lossTenths % 10)) / 10;
}

static void RecordDecision(u8 battler, s32 loss, const struct EvalAction *chosen, const struct EvalAction *best)
{
    u16 lossTenths = GetLossTenths(loss);
    u8 accuracy = GetDecisionAccuracy(loss);
    u8 id = sRoot.active[battler];

    sReport.decisions++;
    sReport.accuracySum += accuracy;
    sReport.counts[GetGrade(lossTenths)]++;
    if (lossTenths > sReport.worstLoss)
    {
        sReport.worstLoss = lossTenths;
        sReport.worstTurn = sWork->turnNumber;
        sReport.worstSpecies = sMons[id].species;
        sReport.worstChosen = *chosen;
        sReport.worstBest = *best;
        sReport.worstChosenMove = chosen->kind == ACT_MOVE ? sMons[id].moves[chosen->index] : MOVE_NONE;
        sReport.worstBestMove = best->kind == ACT_MOVE ? sMons[id].moves[best->index] : MOVE_NONE;
        sReport.worstChosenSwitch = chosen->kind == ACT_SWITCH ? sMons[chosen->index].species : SPECIES_NONE;
        sReport.worstBestSwitch = best->kind == ACT_SWITCH ? sMons[best->index].species : SPECIES_NONE;
    }
    EvalLog("grade", battler, lossTenths, accuracy, sReport.decisions);
}

enum
{
    PHASE_PRUNE,
    PHASE_MATRIX,
    PHASE_SOLVE,
    PHASE_WAIT, // for everyone's choices
    PHASE_GRADE,
    PHASE_DONE,
};

static void StartGrading(void)
{
    u8 b;

    sIsDouble = (gBattleTypeFlags & BATTLE_TYPE_DOUBLE) != 0;
    sWork->turnNumber = gBattleResults.battleTurnCounter + 1;
    BuildTypeChart();
    BuildMons();
    BuildRootState();
    BuildDamageTables();

    for (b = 0; b < MAX_BATTLERS_COUNT; b++)
    {
        sChosen[b].kind = ACT_NONE;
        sChosen[b].index = 0;
        sChosen[b].target = NO_TARGET;
        if (b < gBattlersCount)
        {
            BuildOptions(b);
        }
        else
        {
            sNumOptions[b] = 0;
            AddOption(b, ACT_NONE, 0, NO_TARGET);
            sForced[b] = TRUE;
        }
        GetGreedyAction(&sRoot, b, &sDefault[b]);
    }
    sWork->phase = PHASE_PRUNE;
    sWork->stepBattler = 0;
    sWork->stepIndex = 0;
    sWork->stepOpp = 0;
}

enum
{
    STEP_DONE,     // the phase is finished
    STEP_NEXT,     // bookkeeping only; call again
    STEP_SIMULATE, // sWork->simActs needs simulating
};

static u8 PreparePrune(void)
{
    u8 b = sWork->stepBattler;
    u8 i;

    if (b >= gBattlersCount)
        return STEP_DONE;
    if (sNumOptions[b] <= GetPruneKeep(b) && !(sWork->useAI && sIsDouble && !IsPlayerBattler(b) && sNumOptions[b] > 1))
    {
        sNumPruned[b] = 0;
        for (i = 0; i < sNumOptions[b]; i++)
            sPruned[b][sNumPruned[b]++] = i;
        sWork->stepBattler++;
        return STEP_NEXT;
    }
    for (i = 0; i < MAX_BATTLERS_COUNT; i++)
        sWork->simActs[i] = sDefault[i];
    sWork->simActs[b] = sOptions[b][sWork->stepIndex];
    return STEP_SIMULATE;
}

static void StorePrune(u16 v)
{
    u8 b = sWork->stepBattler;

    sWork->values[b][sWork->stepIndex] = IsPlayerBattler(b) ? v : Q_ONE - v;
    if (++sWork->stepIndex >= sNumOptions[b])
    {
        SelectPrunedOptions(b, GetPruneKeep(b));
        sWork->stepBattler++;
        sWork->stepIndex = 0;
    }
}

static u8 PrepareMatrix(void)
{
    if (sWork->stepIndex >= sWork->numJointPlayer)
        return STEP_DONE;
    FillActions(sWork->simActs, sJointPlayer[sWork->stepIndex], sJointOpponent[sWork->stepOpp]);
    return STEP_SIMULATE;
}

static void StoreMatrix(u16 v)
{
    sMatrix[sWork->stepIndex][sWork->stepOpp] = v;
    if (++sWork->stepOpp >= sWork->numJointOpponent)
    {
        sWork->stepOpp = 0;
        sWork->stepIndex++;
    }
}

static void FinishBattlerGrade(u8 battler)
{
    s32 chosen = FindOption(battler, &sChosen[battler]);
    s32 i, best = -1;

    for (i = 0; i < sNumOptions[battler]; i++)
    {
        u16 value;

        if (sWork->valueWeights[i] == 0)
            continue;
        value = sWork->valueSums[i] / sWork->valueWeights[i];
        sWork->values[battler][i] = value;
        if (best < 0 || value > sWork->values[battler][best])
            best = i;
        EvalLog("option", sOptions[battler][i].kind, sOptions[battler][i].kind == ACT_MOVE ? sMons[sRoot.active[battler]].moves[sOptions[battler][i].index] : sOptions[battler][i].index, sOptions[battler][i].target, value);
    }
    if (best >= 0 && chosen >= 0 && sWork->valueWeights[chosen] != 0)
        RecordDecision(battler, sWork->values[battler][best] - sWork->values[battler][chosen], &sOptions[battler][chosen], &sOptions[battler][best]);
}

static bool8 IsGradable(u8 battler)
{
    return battler < gBattlersCount && IsPlayerBattler(battler) && !sForced[battler]
        && FindOption(battler, &sChosen[battler]) >= 0;
}

// Values the player's options against each likely opponent reply. Each player
// Pokemon is graded with its partner's actual choice held fixed.
static u8 PrepareGrade(void)
{
    u8 b = sWork->stepBattler;
    u8 i = sWork->stepIndex;
    u8 o = sWork->stepOpp;
    u8 partner = b ^ BIT_FLANK;
    const struct EvalAction *opt;

    if (b >= gBattlersCount)
        return STEP_DONE;
    if (!IsGradable(b))
    {
        sWork->stepBattler += 2;
        return STEP_NEXT;
    }
    if (i >= sNumOptions[b])
    {
        FinishBattlerGrade(b);
        sWork->stepBattler += 2;
        sWork->stepIndex = 0;
        sWork->stepOpp = 0;
        return STEP_NEXT;
    }
    if (o == 0)
    {
        sWork->valueSums[i] = 0;
        sWork->valueWeights[i] = 0;
    }
    if (o >= sWork->numJointOpponent)
    {
        sWork->stepIndex++;
        sWork->stepOpp = 0;
        return STEP_NEXT;
    }

    opt = &sOptions[b][i];
    if (sOppStrategy[o] < Q_ONE / 64
     || (sIsDouble && opt->kind == ACT_SWITCH && sChosen[partner].kind == ACT_SWITCH && sChosen[partner].index == opt->index))
    {
        sWork->stepOpp++;
        return STEP_NEXT;
    }
    if (!sIsDouble && i < sWork->numJointPlayer)
    {
        // Every option the engine listed is in the matrix, in order
        sWork->valueSums[i] += sOppStrategy[o] * sMatrix[i][o];
        sWork->valueWeights[i] += sOppStrategy[o];
        sWork->stepOpp++;
        return STEP_NEXT;
    }
    sWork->simActs[1] = sOptions[1][sJointOpponent[o][0]];
    sWork->simActs[b] = *opt;
    if (sIsDouble)
    {
        sWork->simActs[3] = sOptions[3][sJointOpponent[o][1]];
        sWork->simActs[partner] = (sChosen[partner].kind != ACT_NONE) ? sChosen[partner] : sDefault[partner];
    }
    else
    {
        sWork->simActs[2].kind = sWork->simActs[3].kind = ACT_NONE;
    }
    return STEP_SIMULATE;
}

static void StoreGrade(u16 v)
{
    u8 i = sWork->stepIndex;
    u8 o = sWork->stepOpp;

    sWork->valueSums[i] += sOppStrategy[o] * v;
    sWork->valueWeights[i] += sOppStrategy[o];
    sWork->stepOpp++;
}

// --------------------------------------------------------------------------
// Opponent AI
// --------------------------------------------------------------------------

static u8 GetAISkill(void)
{
    if (FlagGet(FLAG_SYS_GAME_CLEAR))
        return 100;
    if ((gBattleTypeFlags & BATTLE_TYPE_TRAINER_TOWER) || gTrainerBattleOpponent_A >= NUM_TRAINERS)
        return 100;
    return sTrainerAISkill[gTrainerBattleOpponent_A];
}

// Picks one of `n` options. `best` is how a perfect player would choose (Q12
// probabilities) and `accuracy` is each option's decision accuracy. The choice
// is drawn from a blend of `best` and uniform (or, below the uniform blend's
// accuracy, of uniform and a lean towards the worst options) that makes the
// expected accuracy equal to the target `skill`.
static u8 PickWithSkill(const u16 *best, const u8 *accuracy, u8 n, u8 skill, u8 excluded)
{
    s32 probs[MAX_OPTIONS];
    s32 eBest = 0, eUnif = 0, eWorst = 0, worstTotal = 0;
    s32 target = skill * 16;
    s32 lambda, total = 0, r;
    u8 i;

    if (n <= 1)
        return 0;
    for (i = 0; i < n; i++)
    {
        eBest += best[i] * accuracy[i];
        eUnif += accuracy[i];
        eWorst += (100 - accuracy[i]) * accuracy[i];
        worstTotal += 100 - accuracy[i];
    }
    eBest >>= 8;                // accuracy * 16
    eUnif = eUnif * 16 / n;
    eWorst = worstTotal ? eWorst * 16 / worstTotal : eUnif;

    for (i = 0; i < n; i++)
    {
        s32 unif = Q_ONE / n;

        if (target >= eUnif)
        {
            lambda = (eBest > eUnif) ? (target - eUnif) * Q_ONE / (eBest - eUnif) : Q_ONE;
            if (lambda > Q_ONE)
                lambda = Q_ONE;
            probs[i] = QM(lambda, best[i]) + QM(Q_ONE - lambda, unif);
        }
        else
        {
            s32 worst = worstTotal ? (100 - accuracy[i]) * Q_ONE / worstTotal : unif;

            lambda = (eUnif > eWorst) ? (eUnif - target) * Q_ONE / (eUnif - eWorst) : 0;
            if (lambda > Q_ONE)
                lambda = Q_ONE;
            probs[i] = QM(lambda, worst) + QM(Q_ONE - lambda, unif);
        }
        if (i == excluded)
            probs[i] = 0;
        total += probs[i];
    }
    if (total <= 0)
        return (excluded == 0 && n > 1) ? 1 : 0;
    r = (Random() * total) >> 16;
    for (i = 0; i < n; i++)
    {
        if (r < probs[i])
            return i;
        r -= probs[i];
    }
    return n - 1;
}

// Decides what each opponent Pokemon does this turn, once the equilibrium is known
static void ChooseAIActions(void)
{
    u8 k;

    sWork->aiReady = TRUE;
    // A perfect player follows the equilibrium mix of joint options
    EvalLog("ai ready", gMain.vblankCounter2, sWork->aiSkill, 0, 0);
    if (sWork->aiSkill >= 100)
    {
        u32 r = (Random() * Q_ONE) >> 16;
        u8 o;

        for (o = 0; o < sWork->numJointOpponent - 1; o++)
        {
            if (r < sOppStrategy[o])
                break;
            r -= sOppStrategy[o];
        }
        sWork->aiAction[1] = sOptions[1][sJointOpponent[o][0]];
        sWork->aiAction[3] = sOptions[3][sJointOpponent[o][1]];
        if (gBattlersCount > 1 && !sForced[1])
        {
            sReport.foeDecisions++;
            sReport.foeAccuracySum += 100;
        }
        if (sIsDouble && !sForced[3])
        {
            sReport.foeDecisions++;
            sReport.foeAccuracySum += 100;
        }
        return;
    }

    for (k = 0; k < (sIsDouble ? 2 : 1); k++)
    {
        u8 b = B_POSITION_OPPONENT_LEFT + 2 * k;
        u16 values[MAX_OPTIONS];
        u16 best[MAX_OPTIONS];
        u8 accuracy[MAX_OPTIONS];
        u8 i, o, n = sNumOptions[b], pick, excluded = 0xFF;
        s32 top = 0;

        sWork->aiAction[b] = sOptions[b][0];
        if (sForced[b] || n <= 1)
            continue;
        for (i = 0; i < n; i++)
        {
            best[i] = 0;
            if (sIsDouble)
            {
                values[i] = sWork->values[b][i];
            }
            else
            {
                // Against the player's equilibrium mix; every option is in the matrix
                s32 v = 0;
                u8 p;

                for (p = 0; p < sWork->numJointPlayer; p++)
                    v += QM(sPlayerStrategy[p], sMatrix[p][i]);
                values[i] = Q_ONE - v;
            }
            if (values[i] > top)
                top = values[i];
        }
        for (o = 0; o < sWork->numJointOpponent; o++)
            best[sJointOpponent[o][k]] += sOppStrategy[o];
        for (i = 0; i < n; i++)
        {
            accuracy[i] = GetDecisionAccuracy(top - values[i]);
            // Partners can't both switch to the same Pokemon
            if (k == 1 && sWork->aiAction[1].kind == ACT_SWITCH && sOptions[b][i].kind == ACT_SWITCH
             && sOptions[b][i].index == sWork->aiAction[1].index)
                excluded = i;
        }
        pick = PickWithSkill(best, accuracy, n, sWork->aiSkill, excluded);
        sWork->aiAction[b] = sOptions[b][pick];
        EvalLog("ai pick", b, sOptions[b][pick].kind == ACT_MOVE ? sMons[sRoot.active[b]].moves[sOptions[b][pick].index] : sOptions[b][pick].index, accuracy[pick], sWork->aiSkill);
        sReport.foeDecisions++;
        sReport.foeAccuracySum += accuracy[pick];
    }
}

// --------------------------------------------------------------------------
// Turn control
// --------------------------------------------------------------------------

enum
{
    STEP_RESULT_BUSY,
    STEP_RESULT_IDLE,     // waiting for everyone's choices
    STEP_RESULT_FINISHED,
};

static void StartGradePhase(void)
{
    sWork->phase = PHASE_GRADE;
    sWork->stepBattler = B_POSITION_PLAYER_LEFT;
    sWork->stepIndex = 0;
    sWork->stepOpp = 0;
}

// Does one step of the turn's evaluation, at most one simulated turn
static u8 EvalStep(void)
{
    u8 o, result;
    u16 v;

    if (sWork->simStage != 0)
    {
        if (ContinueSimulation(&v))
        {
            switch (sWork->phase)
            {
            case PHASE_PRUNE:
                StorePrune(v);
                break;
            case PHASE_MATRIX:
                StoreMatrix(v);
                break;
            case PHASE_GRADE:
                StoreGrade(v);
                break;
            }
        }
        return STEP_RESULT_BUSY;
    }

    switch (sWork->phase)
    {
    case PHASE_PRUNE:
        result = PreparePrune();
        if (result == STEP_DONE)
        {
            sWork->numJointPlayer = BuildJointOptions(B_SIDE_PLAYER, sJointPlayer, MAX_JOINT_PLAYER);
            sWork->numJointOpponent = BuildJointOptions(B_SIDE_OPPONENT, sJointOpponent, MAX_JOINT_OPPONENT);
            if (sWork->numJointPlayer == 0 || sWork->numJointOpponent == 0)
                return STEP_RESULT_FINISHED;
            sWork->phase = PHASE_MATRIX;
            sWork->stepIndex = 0;
            sWork->stepOpp = 0;
        }
        break;
    case PHASE_MATRIX:
        result = PrepareMatrix();
        if (result == STEP_DONE)
            sWork->phase = PHASE_SOLVE;
        break;
    case PHASE_SOLVE:
        SolveEquilibrium(sWork->numJointPlayer, sWork->numJointOpponent);
        for (o = 0; o < sWork->numJointOpponent; o++)
            EvalLog("opp", sJointOpponent[o][0], sJointOpponent[o][1], sOppStrategy[o], 0);
        if (sWork->useAI)
            ChooseAIActions();
        if (sWork->choicesKnown)
            StartGradePhase();
        else
            sWork->phase = PHASE_WAIT;
        return STEP_RESULT_BUSY;
    case PHASE_WAIT:
        return STEP_RESULT_IDLE;
    case PHASE_GRADE:
        result = PrepareGrade();
        if (result == STEP_DONE)
            return STEP_RESULT_FINISHED;
        break;
    default:
        return STEP_RESULT_FINISHED;
    }
    if (result == STEP_SIMULATE)
        BeginSimulation();
    return STEP_RESULT_BUSY;
}

static void EndEvaluation(void)
{
    EvalLog("eval end", gMain.vblankCounter2, 0, 0, 0);
    FREE_AND_SET_NULL(sWork);
}

// Scanlines left before the next VBlank
static s32 GetScanlinesLeft(void)
{
    s32 line = REG_VCOUNT;

    return (line >= DISPLAY_HEIGHT) ? (228 - line + DISPLAY_HEIGHT) : (DISPLAY_HEIGHT - line);
}

// Called every frame of the battle: evaluates in the background, in whatever
// time the frame has to spare. (Not a task, as the bag and party screens reset those.)
void BattleEval_Update(void)
{
    if (sWork == NULL)
        return;
    while (GetScanlinesLeft() > 56)
    {
        u8 result = EvalStep();

        if (result == STEP_RESULT_FINISHED)
        {
            EndEvaluation();
            return;
        }
        if (result == STEP_RESULT_IDLE)
            return;
    }
}

// Runs the evaluation still in progress to the end
static void FinishEvaluation(void)
{
    if (sWork == NULL)
        return;
    while (EvalStep() == STEP_RESULT_BUSY)
        ;
    EndEvaluation();
}

static bool8 ShouldEvaluate(void)
{
    if (!(gBattleTypeFlags & BATTLE_TYPE_TRAINER))
        return FALSE;
    if (gBattleTypeFlags & (BATTLE_TYPE_LINK | BATTLE_TYPE_MULTI | BATTLE_TYPE_SAFARI | BATTLE_TYPE_OLD_MAN_TUTORIAL | BATTLE_TYPE_POKEDUDE))
        return FALSE;
    return TRUE;
}

static bool8 AllocWork(void)
{
    sWork = AllocZeroed(sizeof(struct EvalWork));
    if (sWork == NULL)
        return FALSE;
    sIsDouble = (gBattleTypeFlags & BATTLE_TYPE_DOUBLE) != 0;
    return TRUE;
}

void BattleEval_Reset(void)
{
    s32 i;

    sReport.decisions = 0;
    sReport.accuracySum = 0;
    sReport.foeDecisions = 0;
    sReport.foeAccuracySum = 0;
    for (i = 0; i < EVAL_GRADE_COUNT; i++)
        sReport.counts[i] = 0;
    sReport.worstLoss = 0;
    sReport.worstTurn = 0;
    gEvalReportReady = FALSE;
    // Left over from a battle that ended abruptly
    TRY_FREE_AND_SET_NULL(sWork);
}

// Called as the battlers start choosing their actions for a turn. The opponent
// thinks while the player chooses.
void BattleEval_StartTurn(void)
{
    FinishEvaluation();
    if (!ShouldEvaluate() || !AllocWork())
        return;
    EvalLog("eval start", gMain.vblankCounter2, sizeof(struct EvalWork), 0, 0);
    sWork->useAI = TRUE;
    sWork->aiSkill = GetAISkill();
    StartGrading();
}

// Called once every battler has chosen: grades the player's choices
void BattleEval_GradeTurn(void)
{
    u8 b;

    if (sWork == NULL)
        return;
    for (b = 0; b < gBattlersCount; b++)
    {
        GetChosenAction(b, &sChosen[b]);
        // Moves the player picked that the engine wouldn't list, such as hitting a partner
        if (IsPlayerBattler(b) && sChosen[b].kind != ACT_NONE && FindOption(b, &sChosen[b]) < 0)
            AddOption(b, sChosen[b].kind, sChosen[b].index, sChosen[b].target);
    }
    sWork->choicesKnown = TRUE;
    if (sWork->phase == PHASE_WAIT)
        StartGradePhase();
}

bool8 BattleEval_IsDecidingForOpponent(void)
{
    return sWork != NULL && sWork->useAI;
}

// Gets the engine's choice for an opponent battler. Returns FALSE while it is still thinking.
bool8 BattleEval_GetOpponentAction(u8 battler, u8 *kind, u8 *index, u8 *target)
{
    const struct EvalAction *act;

    if (sWork == NULL || !sWork->aiReady)
        return FALSE;
    act = &sWork->aiAction[battler];
    *target = act->target;
    switch (act->kind)
    {
    case ACT_SWITCH:
        *kind = EVAL_ACTION_SWITCH;
        *index = act->index - PARTY_SIZE;
        break;
    case ACT_MOVE:
        *kind = EVAL_ACTION_MOVE;
        *index = act->index;
        break;
    default:
        *kind = EVAL_ACTION_NONE;
        *index = 0;
        break;
    }
    return TRUE;
}

// Picks which Pokemon an opponent sends out after one faints, by playing out
// a couple of turns with each candidate. Returns PARTY_SIZE to leave it to the old AI.
u8 BattleEval_ChooseReplacement(u8 battler)
{
    u16 values[PARTY_SIZE], best[PARTY_SIZE];
    u8 accuracy[PARTY_SIZE], candidates[PARTY_SIZE];
    u8 n = 0, i, b, pick, partnerPick;
    s32 top = 0;

    FinishEvaluation();
    if (!ShouldEvaluate() || !AllocWork())
        return PARTY_SIZE;
    sWork->turnNumber = gBattleResults.battleTurnCounter + 1;
    BuildTypeChart();
    BuildMons();
    BuildRootState();
    BuildDamageTables();

    partnerPick = sIsDouble ? gBattleStruct->monToSwitchIntoId[battler ^ BIT_FLANK] : PARTY_SIZE;
    for (i = 0; i < PARTY_SIZE; i++)
    {
        u8 id = PARTY_SIZE + i;
        struct EvalAction acts[MAX_BATTLERS_COUNT];
        s32 material;
        u8 turn;

        if (!sMons[id].present || sRoot.mons[id].alive == 0 || IsActive(&sRoot, id) || i == partnerPick)
            continue;
        sSim = sRoot;
        DoSwitch(&sSim, battler, id);
        material = GetMaterial(&sSim);
        for (turn = 0; turn < 2; turn++)
        {
            for (b = 0; b < gBattlersCount; b++)
                GetRolloutAction(&sSim, b, &acts[b]);
            SimulateTurn(&sSim, acts);
            material += GetMaterial(&sSim);
        }
        values[n] = Q_ONE - GetWinProb(material / 3);
        if (values[n] > top)
            top = values[n];
        candidates[n++] = i;
    }
    if (n == 0)
    {
        FREE_AND_SET_NULL(sWork);
        return PARTY_SIZE;
    }

    // A perfect player splits evenly between the best candidates
    for (i = 0, b = 0; i < n; i++)
    {
        accuracy[i] = GetDecisionAccuracy(top - values[i]);
        if (accuracy[i] == 100)
            b++;
    }
    for (i = 0; i < n; i++)
        best[i] = (accuracy[i] == 100) ? Q_ONE / b : 0;
    pick = PickWithSkill(best, accuracy, n, GetAISkill(), 0xFF);
    sReport.foeDecisions++;
    sReport.foeAccuracySum += accuracy[pick];
    FREE_AND_SET_NULL(sWork);
    return candidates[pick];
}

void BattleEval_BuildReport(void)
{
    u8 *s = gEvalReportText;
    u16 accuracy;

    FinishEvaluation();
    gEvalReportReady = FALSE;
    if (sReport.decisions == 0)
        return;
    accuracy = (sReport.accuracySum + sReport.decisions / 2) / sReport.decisions;

    s = StringCopy(s, sText_DecisionAccuracy);
    s = ConvertIntToDecimalStringN(s, accuracy, STR_CONV_MODE_LEFT_ALIGN, 3);
    s = StringCopy(s, sText_Best);
    s = ConvertIntToDecimalStringN(s, sReport.counts[EVAL_GRADE_BEST], STR_CONV_MODE_LEFT_ALIGN, 3);
    s = StringCopy(s, sText_Good);
    s = ConvertIntToDecimalStringN(s, sReport.counts[EVAL_GRADE_GOOD], STR_CONV_MODE_LEFT_ALIGN, 3);
    s = StringCopy(s, sText_Inaccurate);
    s = ConvertIntToDecimalStringN(s, sReport.counts[EVAL_GRADE_INACCURACY], STR_CONV_MODE_LEFT_ALIGN, 3);
    s = StringCopy(s, sText_Mistakes);
    s = ConvertIntToDecimalStringN(s, sReport.counts[EVAL_GRADE_MISTAKE], STR_CONV_MODE_LEFT_ALIGN, 3);
    s = StringCopy(s, sText_Blunders);
    s = ConvertIntToDecimalStringN(s, sReport.counts[EVAL_GRADE_BLUNDER], STR_CONV_MODE_LEFT_ALIGN, 3);
    if (sReport.foeDecisions != 0)
    {
        s = StringCopy(s, sText_FoeAccuracy);
        s = ConvertIntToDecimalStringN(s, (sReport.foeAccuracySum + sReport.foeDecisions / 2) / sReport.foeDecisions, STR_CONV_MODE_LEFT_ALIGN, 3);
        s = StringCopy(s, sText_Percent);
    }

    // Point out the costliest decision, if it was at least an inaccuracy
    if (sReport.worstLoss > sGradeThresholds[EVAL_GRADE_BEST + 1])
    {
        s = StringCopy(s, sText_BiggestSlip);
        s = ConvertIntToDecimalStringN(s, sReport.worstTurn, STR_CONV_MODE_LEFT_ALIGN, 3);
        s = StringCopy(s, sText_Colon);
        s = StringCopy(s, gSpeciesNames[sReport.worstSpecies]);
        if (sReport.worstChosen.kind == ACT_SWITCH)
        {
            s = StringCopy(s, sText_SwitchedTo);
            s = StringCopy(s, gSpeciesNames[sReport.worstChosenSwitch]);
        }
        else
        {
            s = StringCopy(s, sText_Used);
            s = StringCopy(s, gMoveNames[sReport.worstChosenMove]);
        }
        s = StringCopy(s, sText_Comma);
        if (sReport.worstBest.kind == ACT_SWITCH)
        {
            s = StringCopy(s, sText_SwitchingTo);
            s = StringCopy(s, gSpeciesNames[sReport.worstBestSwitch]);
        }
        else if (sReport.worstBest.kind == ACT_MOVE)
        {
            s = StringCopy(s, sText_Using);
            s = StringCopy(s, gMoveNames[sReport.worstBestMove]);
        }
        else
        {
            s = StringCopy(s, sText_HoldingBack);
        }
        s = StringCopy(s, sText_WasBetter);
    }
    gEvalReportReady = TRUE;
}
