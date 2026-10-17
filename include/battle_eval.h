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

extern u8 gEvalReportReady;
extern u8 gEvalReportText[];

void BattleEval_Reset(void);
void BattleEval_GradeTurn(void);
void BattleEval_BuildReport(void);

#endif // GUARD_BATTLE_EVAL_H
