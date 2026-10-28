# A roll from 14-16 June 1987; Vision misreads one stamp's day (7-segment 4 -> 1) cleanly.
# True date of roll_03 is 1987-06-14; Vision reads '87 6 11 on every variant. Model never asked.
FILES = {
    'roll/roll_01.jpg': dict(v=["'87 6 14"]),
    'roll/roll_02.jpg': dict(v=["'87 6 14"]),
    'roll/roll_03.jpg': dict(v=["'87 6 11"]),   # truth '87 6 14
    'roll/roll_04.jpg': dict(v=["'87 6 16"]),
    'roll/roll_05.jpg': dict(v=["'87 6 16"]),
    'roll/roll_06.jpg': dict(v=["'87 8 16"]),   # truth '87 6 16: 6->8 misread (60 days later)... out of NEAR? 61 days
}
