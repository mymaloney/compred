#ifndef GUARD_BATTLE_EVAL_H
#define GUARD_BATTLE_EVAL_H

// Decision grades, from best to worst.
enum
{
    EVAL_GRADE_BEST,
    EVAL_GRADE_GOOD,
    EVAL_GRADE_INACCURACY,
    EVAL_GRADE_MISTAKE,
    EVAL_GRADE_BLUNDER,
    EVAL_GRADE_COUNT,
};

// What the engine decided for an opponent battler
enum
{
    EVAL_ACTION_NONE,
    EVAL_ACTION_MOVE,
    EVAL_ACTION_SWITCH,
};

extern u8 gEvalReportReady;
extern u8 gEvalReportText[];

void BattleEval_Reset(void);
void BattleEval_StartTurn(void);
void BattleEval_Update(void);
void BattleEval_GradeTurn(void);
bool8 BattleEval_IsDecidingForOpponent(void);
bool8 BattleEval_GetOpponentAction(u8 battler, u8 *kind, u8 *index, u8 *target);
u8 BattleEval_ChooseReplacement(u8 battler);
void BattleEval_BuildReport(void);

#endif // GUARD_BATTLE_EVAL_H
