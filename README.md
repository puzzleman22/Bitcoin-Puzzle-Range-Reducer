# Bitcoin Puzzle GPU Range Reducer

<img src="https://raw.githubusercontent.com/puzzleman22/Bitcoin-Puzzle-Range-Reducer/refs/heads/main/BPRD.png" />

## what its about?

this program gets any range that starts with 1, for example puzzle 21

1 prefix + 5 hex chars

so if you call

`main.exe 0 29a78213caa9eea824acf08022ab9dfc83414f56`

this means that the first 5 hex chars will be fully scanned in sequential mode, after the prefix "1"

the second parameter 0 - is the amount of chars to generate randomly 

so for puzzle 73 it will be

`main.exe 13 105b7f253f0ebd7843adaebbd805c944bfb863e4`

1 prefix + 13 random hex chars + 5 fully scanned

this way you only need to randomize a less amount of hex chars to hit the target

## How to use

`main.exe 0 29a78213caa9eea824acf08022ab9dfc83414f56` - puzzle 21
`main.exe 1 2f396b29b27324300d0c59b17c3abc1835bd3dbb` - puzzle 25
`main.exe 13 105b7f253f0ebd7843adaebbd805c944bfb863e4` - puzzle 73
