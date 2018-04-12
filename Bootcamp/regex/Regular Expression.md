

```python
import re
```

## Simple Python Match


```python
# matching string
pattern1 = 'cat'
pattern2 = 'bird'
string = 'dog runs to cat'
print(pattern1 in string)
print(pattern2 in string)
```

    True
    False


## Match with Regex


```python
# matching string
pattern1 = 'cat'
pattern2 = 'bird'
string = 'dog runs to cat'
print(re.search(pattern1, string))
print(re.search(pattern2, string))
```

    <_sre.SRE_Match object; span=(12, 15), match='cat'>
    None


## Match with Multiple Patterns using []


```python
# mutiple patterns ("run" or "ran")
ptn = r"r[au]n"
print(re.search(ptn, 'dog runs to cat'))
```

    <_sre.SRE_Match object; span=(4, 7), match='run'>


## Useful Pattern Matching


```python
# continue
print(re.search(r"r[A-Z]n", 'dog runs to cat'))
print(re.search(r"r[a-z]n", 'dog runs to cat'))
print(re.search(r"r[0-9]n", 'dog r2ns to cat'))
print(re.search(r"r[0-9a-z]n", 'dog runs to cat'))
```

    None
    <_sre.SRE_Match object; span=(4, 7), match='run'>
    <_sre.SRE_Match object; span=(4, 7), match='r2n'>
    <_sre.SRE_Match object; span=(4, 7), match='run'>


## Special Type Matching

### Numbers


```python
# \d : decimal digit
print(re.search(r"r\dn", 'run r4n'))
# \D : any non-decimal digit
print(re.search(r"r\Dn", 'run r4n'))
```

    <_sre.SRE_Match object; span=(4, 7), match='r4n'>
    <_sre.SRE_Match object; span=(0, 3), match='run'>


### White space


```python
# \s : any white space [\t\n\r\f\v]
print(re.search(r"r\sn", 'r\nn r4n'))
# \S : opposite to \s, any non-white space
print(re.search(r"r\Sn", 'r\nn r4n'))
```

    <_sre.SRE_Match object; span=(0, 3), match='r\nn'>
    <_sre.SRE_Match object; span=(4, 7), match='r4n'>


### All Numbers, Letters and "_"


```python
# \w : [a-zA-z0-9_]
print(re.search(r"r\wn", 'r\nn r4n'))
# \W : opposite to \w
print(re.search(r"r\Wn", 'r\nn r4n'))
```

    <_sre.SRE_Match object; span=(4, 7), match='r4n'>
    <_sre.SRE_Match object; span=(0, 3), match='r\nn'>


### Empty String


```python
# \b : empty string (only at the start or end of the word)
print(re.search(r"\bruns\b", 'dog runs to cat'))
# \B : empty string (but not at the start or end of a word)
print(re.search(r"\B runs \B", 'dog  runs  to cat'))
```

    <_sre.SRE_Match object; span=(4, 8), match='runs'>
    <_sre.SRE_Match object; span=(4, 10), match=' runs '>


### Special Characters


```python
# \\ : match \
print(re.search(r"runs\\", 'runs\ to me'))
# . : match anything (except \n)
print(re.search(r"r.n", 'r[ns to me'))
```

    <_sre.SRE_Match object; span=(0, 5), match='runs\\'>
    <_sre.SRE_Match object; span=(0, 3), match='r[n'>


### Start and End


```python
# ^ : match line beginning 
print(re.search(r"^dog", 'dog runs to dog'))
# $ : match line ending
print(re.search(r"cat$", 'cat runs to cat'))
```

    <_sre.SRE_Match object; span=(0, 3), match='dog'>
    <_sre.SRE_Match object; span=(12, 15), match='cat'>


### Maybe


```python
# ? : may or may not occur
print(re.search(r"Mon(day)?", 'Monday'))
print(re.search(r"Mon(day)?", 'Mon'))
```

    <_sre.SRE_Match object; span=(0, 6), match='Monday'>
    <_sre.SRE_Match object; span=(0, 3), match='Mon'>


## Multi-Line Matching


```python
# multi-line
string = """
dog runs to cat.
I run to dog.
"""
print(re.search(r"^I", string))
print(re.search(r"^I", string, flags=re.M))
```

    None
    <_sre.SRE_Match object; span=(18, 19), match='I'>


## Match 0 or Multiple times


```python
# * : occur 0 or more times
print(re.search(r"ab*", 'a'))
print(re.search(r"ab*", 'abbbbbb'))
```

    <_sre.SRE_Match object; span=(0, 1), match='a'>
    <_sre.SRE_Match object; span=(0, 7), match='abbbbbb'>


## Match 1 or Multiple times


```python
# + : occur 1 or more times
print(re.search(r"ab+", 'a'))
print(re.search(r"ab+", 'abbbbbb'))
```

    None
    <_sre.SRE_Match object; span=(0, 7), match='abbbbbb'>


## Defined Match Times


```python
# {n, m} : occur n to m time
print(re.search(r"ab{2,10}", 'a'))
print(re.search(r"ab{2,10}", 'abbbbbb'))
```

    None
    <_sre.SRE_Match object; span=(0, 7), match='abbbbbb'>


## Group


```python
# group
match = re.search(r"(\d+), Date: (.+)", 'ID: 021523, Date: Feb/12/2017')
print(match.group())
print(match.group(1))
print(match.group(2))
```

    021523, Date: Feb/12/2017
    021523
    Feb/12/2017



```python
match = re.search(r"(?P<id>\d+), Date: (?P<date>.+)", 'ID: 025123, Date: Feb/12/2017')
print(match.group('id'))
print(match.group('date'))
```

    025123
    Feb/12/2017


## Find All Matchs


```python
# find all
print(re.findall(r"r[ua]n", 'run ran ren'))
```

    ['run', 'ran']



```python
# | : or
print(re.findall(r"(run|ran)", 'run ran ren'))
```

    ['run', 'ran']


## Replace


```python
# re.sub() : replace
print(re.sub(r"r[au]ns", 'catches', 'dog runs to cat'))
```

    dog catches to cat


## Split


```python
# re.split()
print(re.split(r"[,;\.]", 'a;b,c.d;e'))
```

    ['a', 'b', 'c', 'd', 'e']


## Compile


```python
# compile
compiled_re = re.compile(r"r[au]n")
print(compiled_re.search('dog runs to cat'))
```

    <_sre.SRE_Match object; span=(4, 7), match='run'>

