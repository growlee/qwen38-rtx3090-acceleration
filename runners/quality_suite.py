"""Pre-execution synthetic quality fixtures for the Qwen3.8 comparison.

This module contains fixtures only.  It never calls a model and never executes
candidate-produced code.  ``reference_source`` is trusted suite-owned code used
to audit literal expected values. Runtime generation controls, candidate-code
sandboxing, and transport remain separate preconditions for freezing a run.
"""

from __future__ import annotations

import copy


SCHEMA_VERSION = 1
SUITE_ID = "qwen38-quality-60-v1"


def _request(prompt: str, seed: int, max_tokens: int = 4096) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only a Python function definition with the requested name. "
                    "Do not use markdown, imports, input/output, network, files, or global state."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "generation": {"temperature": 0, "seed": seed, "max_tokens": max_tokens},
    }


def _coding(case_id: str, function_name: str, prompt: str, source: str, cases: list[dict], seed: int) -> dict:
    return {
        "case_id": case_id,
        "category": "coding",
        "critical": True,
        "request": _request(prompt, seed),
        "check": {"type": "python_function_cases", "function_name": function_name, "cases": cases},
        "reference_source": source,
    }


def _fc(args: list, expected, kwargs: dict | None = None) -> dict:
    return {"args": args, "kwargs": kwargs or {}, "expected": expected}


def _fe(args: list, expected_exception: str = "ValueError", kwargs: dict | None = None) -> dict:
    return {"args": args, "kwargs": kwargs or {}, "expected_exception": expected_exception}


_CODING = [
    _coding("code-clamp", "clamp", "Define clamp(value, low, high). Raise ValueError if low > high; otherwise bound value inclusively.",
            "def clamp(value, low, high):\n    if low > high: raise ValueError('low > high')\n    return min(high, max(low, value))",
            [_fc([5, 0, 10], 5), _fc([-2, 0, 10], 0), _fc([12, 0, 10], 10), _fc([0, 0, 0], 0), _fc([2.5, 1.5, 2.0], 2.0), _fe([1, 3, 2])], 8201),
    _coding("code-dedupe-preserve", "dedupe_preserve", "Define dedupe_preserve(items) returning the first occurrence of each hashable item, in original order.",
            "def dedupe_preserve(items):\n    seen=set(); out=[]\n    for x in items:\n        if x not in seen: seen.add(x); out.append(x)\n    return out",
            [_fc([[]], []), _fc([[1,1,2,1]], [1,2]), _fc([["b","a","b"]], ["b","a"]), _fc([[0,False,1]], [0,1]), _fc([["xy","xy","yx"]], ["xy","yx"])], 8202),
    _coding("code-chunked", "chunked", "Define chunked(items, size) returning consecutive list chunks. Raise ValueError when size <= 0.",
            "def chunked(items, size):\n    if size <= 0: raise ValueError('size')\n    return [list(items[i:i+size]) for i in range(0,len(items),size)]",
            [_fc([[],3], []), _fc([[1],2], [[1]]), _fc([[1,2,3,4,5],2], [[1,2],[3,4],[5]]), _fc(["abcd",3], [["a","b","c"],["d"]]), _fc([[1,2,3],1], [[1],[2],[3]]), _fe([[1,2],0])], 8203),
    _coding("code-merge-intervals", "merge_intervals", "Define merge_intervals(intervals) for [start,end] pairs. Sort and merge overlapping or touching intervals; do not mutate input.",
            "def merge_intervals(intervals):\n    out=[]\n    for a,b in sorted((x[0],x[1]) for x in intervals):\n        if not out or a > out[-1][1]: out.append([a,b])\n        else: out[-1][1]=max(out[-1][1],b)\n    return out",
            [_fc([[]], []), _fc([[[1,3]]], [[1,3]]), _fc([[[1,3],[2,6],[8,10]]], [[1,6],[8,10]]), _fc([[[5,7],[1,2],[2,4]]], [[1,4],[5,7]]), _fc([[[-3,-1],[-2,2],[9,9]]], [[-3,2],[9,9]])], 8204),
    _coding("code-balanced-brackets", "balanced_brackets", "Define balanced_brackets(text). Ignore non-bracket characters; validate (), [], and {} nesting.",
            "def balanced_brackets(text):\n    pairs={')':'(',']':'[','}':'{'}; stack=[]\n    for c in text:\n        if c in '([{': stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop()!=pairs[c]: return False\n    return not stack",
            [_fc([""], True), _fc(["a(b[c]{d})"], True), _fc(["([)]"], False), _fc(["(()"], False), _fc(["x}y"], False), _fc(["{[()()]}"] , True)], 8205),
    _coding("code-slugify-ascii", "slugify_ascii", "Define slugify_ascii(text): lowercase ASCII letters/digits, replace each run of other characters with one hyphen, strip edge hyphens.",
            "def slugify_ascii(text):\n    out=[]; gap=False\n    for c in text.lower():\n        if ('a'<=c<='z') or ('0'<=c<='9'):\n            if gap and out: out.append('-')\n            out.append(c); gap=False\n        else: gap=True\n    return ''.join(out)",
            [_fc(["Hello, World!"], "hello-world"), _fc(["  A__B  "], "a-b"), _fc(["123"], "123"), _fc(["---"], ""), _fc(["Café déjà"], "caf-d-j"), _fc(["a---b---c"], "a-b-c"), _fc(["x٢y"], "x-y")], 8206),
    _coding("code-flatten-once", "flatten_once", "Define flatten_once(groups), returning a new list containing each iterable group's elements in order.",
            "def flatten_once(groups):\n    return [x for group in groups for x in group]",
            [_fc([[]], []), _fc([[[1,2],[],[3]]], [1,2,3]), _fc([["ab","c"]], ["a","b","c"]), _fc([[[1],[2,3]]], [1,2,3]), _fc([[[None],[False,0]]], [None,False,0])], 8207),
    _coding("code-transpose-ragged", "transpose_ragged", "Define transpose_ragged(rows, fill=None), padding shorter rows to the maximum width. Empty input returns [].",
            "def transpose_ragged(rows, fill=None):\n    width=max((len(r) for r in rows), default=0)\n    return [[r[i] if i<len(r) else fill for r in rows] for i in range(width)]",
            [_fc([[]], []), _fc([[[1,2],[3,4]]], [[1,3],[2,4]]), _fc([[[1],[2,3]]], [[1,2],[None,3]]), _fc([[[1,2],[]], 0], [[1,0],[2,0]]), _fc([[[],[]]], [])], 8208),
    _coding("code-parse-duration", "parse_duration", "Define parse_duration(text) for a nonempty sequence of integer components suffixed h, m, or s. Units may appear once in any order. Return total seconds; raise ValueError for malformed input.",
            "def parse_duration(text):\n    if not text: raise ValueError('empty')\n    i=0; total=0; seen=set(); mult={'h':3600,'m':60,'s':1}\n    while i<len(text):\n        j=i\n        while j<len(text) and text[j].isdigit(): j+=1\n        if j==i or j==len(text) or text[j] not in mult or text[j] in seen: raise ValueError('bad')\n        seen.add(text[j]); total+=int(text[i:j])*mult[text[j]]; i=j+1\n    return total",
            [_fc(["1h2m3s"],3723), _fc(["45s"],45), _fc(["2m1h"],3720), _fc(["0s"],0), _fc(["10h"],36000), _fe(["1h2h"])], 8209),
    _coding("code-roman-to-int", "roman_to_int", "Define roman_to_int(text) for canonical uppercase Roman numerals I..MMMCMXCIX. Raise ValueError for empty or noncanonical input.",
            "def roman_to_int(text):\n    vals={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    if not text or any(c not in vals for c in text): raise ValueError('bad')\n    n=sum((-vals[c] if i+1<len(text) and vals[c]<vals[text[i+1]] else vals[c]) for i,c in enumerate(text))\n    table=((1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')); x=n; canon=''\n    for v,s in table: canon+=s*(x//v); x%=v\n    if canon!=text or not 1<=n<=3999: raise ValueError('noncanonical')\n    return n",
            [_fc(["I"],1), _fc(["IV"],4), _fc(["XLII"],42), _fc(["MCMXCIV"],1994), _fc(["MMMCMXCIX"],3999), _fe(["IIII"])], 8210),
    _coding("code-top-k-frequent", "top_k_frequent", "Define top_k_frequent(items, k). Return up to k distinct items sorted by descending frequency, breaking ties by first occurrence. k <= 0 returns []. Items are hashable.",
            "def top_k_frequent(items,k):\n    count={}; first={}\n    for i,x in enumerate(items): count[x]=count.get(x,0)+1; first.setdefault(x,i)\n    return sorted(count,key=lambda x:(-count[x],first[x]))[:max(0,k)]",
            [_fc([["a","b","a","c","b","a"],2],["a","b"]), _fc([[1,2,2,1],2],[1,2]), _fc([[],3],[]), _fc([[1,1],0],[]), _fc([[3,2,1],9],[3,2,1])], 8211),
    _coding("code-rotate", "rotate", "Define rotate(items, steps), returning a list rotated right by steps; negative steps rotate left. Do not mutate input.",
            "def rotate(items,steps):\n    xs=list(items)\n    if not xs: return []\n    n=steps%len(xs); return xs[-n:]+xs[:-n] if n else xs",
            [_fc([[],3],[]), _fc([[1,2,3],1],[3,1,2]), _fc([[1,2,3],-1],[2,3,1]), _fc([[1,2,3],3],[1,2,3]), _fc(["abcd",6],["c","d","a","b"])], 8212),
    _coding("code-moving-average", "moving_average", "Define moving_average(values, window) returning arithmetic means for every full window. Raise ValueError if window <= 0; return [] if window exceeds length.",
            "def moving_average(values,window):\n    if window<=0: raise ValueError('window')\n    out=[]; total=0\n    for i,x in enumerate(values):\n        total+=x\n        if i>=window: total-=values[i-window]\n        if i>=window-1: out.append(total/window)\n    return out",
            [_fc([[1,2,3,4],2],[1.5,2.5,3.5]), _fc([[5],1],[5.0]), _fc([[],1],[]), _fc([[1,2],3],[]), _fc([[-1,1,-1,1],2],[0.0,0.0,0.0]), _fe([[1,2],0])], 8213),
    _coding("code-rle-encode", "rle_encode", "Define rle_encode(items), returning [[value,count], ...] for consecutive equal runs.",
            "def rle_encode(items):\n    out=[]\n    for x in items:\n        if out and out[-1][0]==x: out[-1][1]+=1\n        else: out.append([x,1])\n    return out",
            [_fc([[]],[]), _fc([[1]],[ [1,1] ]), _fc([[1,1,2,2,2,1]],[[1,2],[2,3],[1,1]]), _fc(["aaabb"],[["a",3],["b",2]]), _fc([[None,None,False]],[[None,2],[False,1]])], 8214),
    _coding("code-rle-decode", "rle_decode", "Define rle_decode(runs), expanding [value,count] pairs. Raise ValueError when a count is negative or not an integer.",
            "def rle_decode(runs):\n    out=[]\n    for value,count in runs:\n        if type(count) is not int or count<0: raise ValueError('count')\n        out.extend([value]*count)\n    return out",
            [_fc([[]],[]), _fc([[["a",3],["b",2]]],["a","a","a","b","b"]), _fc([[[1,0],[2,1]]],[2]), _fc([[[None,2]]],[None,None]), _fc([[[False,2],[0,1]]],[False,False,0]), _fe([[["x",-1]]])], 8215),
    _coding("code-normalize-posix", "normalize_posix", "Define normalize_posix(path). Collapse empty and '.' segments, resolve '..' without going above root for absolute paths, and preserve leading slash. Empty relative result is '.'.",
            "def normalize_posix(path):\n    absolute=path.startswith('/'); parts=[]\n    for p in path.split('/'):\n        if not p or p=='.': continue\n        if p=='..':\n            if parts and parts[-1]!='..': parts.pop()\n            elif not absolute: parts.append('..')\n        else: parts.append(p)\n    result='/'.join(parts)\n    return ('/'+result if absolute else result or '.')",
            [_fc(["a/./b//c"],"a/b/c"), _fc(["a/b/../c"],"a/c"), _fc(["/../../a"],"/a"), _fc(["../a"],"../a"), _fc([""],"."), _fc(["/"],"/")], 8216),
    _coding("code-word-histogram", "word_histogram", "Define word_histogram(text). A word is a maximal alphanumeric Unicode run. Count case-insensitively and return a dict.",
            "def word_histogram(text):\n    out={}; word=[]\n    for c in text.casefold()+' ':\n        if c.isalnum(): word.append(c)\n        elif word:\n            w=''.join(word); out[w]=out.get(w,0)+1; word=[]\n    return out",
            [_fc([""],{}), _fc(["One, one TWO"],{"one":2,"two":1}), _fc(["Привет-мир привет"],{"привет":2,"мир":1}), _fc(["a1 A1 b2"],{"a1":2,"b2":1}), _fc(["...x..."],{"x":1})], 8217),
    _coding("code-palindrome-normalized", "is_palindrome_normalized", "Define is_palindrome_normalized(text), ignoring non-alphanumeric characters and letter case.",
            "def is_palindrome_normalized(text):\n    s=''.join(c.casefold() for c in text if c.isalnum()); return s==s[::-1]",
            [_fc([""],True), _fc(["A man, a plan, a canal: Panama!"],True), _fc(["race a car"],False), _fc(["А роза упала на лапу Азора"],True), _fc(["12321"],True)], 8218),
    _coding("code-safe-get", "safe_get", "Define safe_get(data, path, default=None). Traverse dict keys and list integer indexes from path; return default on missing key, invalid index/type, or incompatible container.",
            "def safe_get(data,path,default=None):\n    cur=data\n    for key in path:\n        try:\n            if isinstance(cur,dict): cur=cur[key]\n            elif isinstance(cur,list) and type(key) is int: cur=cur[key]\n            else: return default\n        except (KeyError,IndexError): return default\n    return cur",
            [_fc([{"a":{"b":3}},["a","b"]],3), _fc([{"a":[10,20]},["a",1]],20), _fc([{},["x"],"no"],"no"), _fc([[1],[-1]],1), _fc([None,[],9],None), _fc([[1],[3],0],0)], 8219),
    _coding("code-group-pairs", "group_pairs", "Define group_pairs(pairs), returning a dict from each key to a list of its values in encounter order.",
            "def group_pairs(pairs):\n    out={}\n    for k,v in pairs: out.setdefault(k,[]).append(v)\n    return out",
            [_fc([[]],{}), _fc([[['a',1],['b',2],['a',3]]],{'a':[1,3],'b':[2]}), _fc([[['x',1],['x',2]]],{'x':[1,2]}), _fc([[['a',None]]],{'a':[None]}), _fc([[['',1],['z',2]]],{'':[1],'z':[2]})], 8220),
    _coding("code-bisect-left", "bisect_left", "Define bisect_left(sorted_items, value), returning the leftmost insertion index using binary search.",
            "def bisect_left(sorted_items,value):\n    lo=0; hi=len(sorted_items)\n    while lo<hi:\n        mid=(lo+hi)//2\n        if sorted_items[mid]<value: lo=mid+1\n        else: hi=mid\n    return lo",
            [_fc([[],5],0), _fc([[1,3,5],0],0), _fc([[1,3,5],4],2), _fc([[1,3,3,3,5],3],1), _fc([[1,3,5],9],3)], 8221),
    _coding("code-deep-merge", "deep_merge", "Define deep_merge(left, right). Recursively merge only when both values are dicts; otherwise right replaces left. Return new dicts without mutating inputs.",
            "def deep_merge(left,right):\n    out={k:(deep_merge(v,{}) if isinstance(v,dict) else v) for k,v in left.items()}\n    for k,v in right.items():\n        if k in out and isinstance(out[k],dict) and isinstance(v,dict): out[k]=deep_merge(out[k],v)\n        else: out[k]=deep_merge(v,{}) if isinstance(v,dict) else v\n    return out",
            [_fc([{},{}],{}), _fc([{'a':1},{'b':2}],{'a':1,'b':2}), _fc([{'a':{'x':1}},{'a':{'y':2}}],{'a':{'x':1,'y':2}}), _fc([{'a':{'x':1}},{'a':9}],{'a':9}), _fc([{'a':[1]},{'a':[2]}],{'a':[2]})], 8222),
    _coding("code-mask-email", "mask_email", "Define mask_email(email). Require exactly one @, nonempty local/domain. Keep first local character, replace remaining local characters with *, preserve domain; raise ValueError if invalid.",
            "def mask_email(email):\n    if email.count('@')!=1: raise ValueError('email')\n    local,domain=email.split('@')\n    if not local or not domain: raise ValueError('email')\n    return local[0]+'*'*(len(local)-1)+'@'+domain",
            [_fc(["a@example.com"],"a@example.com"), _fc(["alice@example.com"],"a****@example.com"), _fc(["XY@d"],"X*@d"), _fc(["юзер@пример.рф"],"ю***@пример.рф"), _fc(["1@x"],"1@x"), _fe(["missing-at"])], 8223),
    _coding("code-compress-ranges", "compress_ranges", "Define compress_ranges(values) for integers: deduplicate and sort, then return inclusive [start,end] ranges of consecutive values.",
            "def compress_ranges(values):\n    xs=sorted(set(values)); out=[]\n    for x in xs:\n        if not out or x>out[-1][1]+1: out.append([x,x])\n        else: out[-1][1]=x\n    return out",
            [_fc([[]],[]), _fc([[1]],[[1,1]]), _fc([[1,2,3,5,7,8]],[[1,3],[5,5],[7,8]]), _fc([[3,2,2,1]],[[1,3]]), _fc([[-2,-1,1]], [[-2,-1],[1,1]])], 8224),
    _coding("code-round-robin", "round_robin", "Define round_robin(groups), interleaving lists by index while skipping exhausted lists.",
            "def round_robin(groups):\n    out=[]; width=max((len(g) for g in groups),default=0)\n    for i in range(width):\n        for g in groups:\n            if i<len(g): out.append(g[i])\n    return out",
            [_fc([[]],[]), _fc([[[1,2],[3,4]]],[1,3,2,4]), _fc([[[1,2,3],[4],[],[5,6]]],[1,4,5,2,6,3]), _fc([[['a'],['b','c']]],["a","b","c"]), _fc([[[],[0]]],[0])], 8225),
    _coding("code-common-prefix", "longest_common_prefix", "Define longest_common_prefix(strings), returning the longest exact prefix shared by all strings. Empty input returns ''.",
            "def longest_common_prefix(strings):\n    if not strings: return ''\n    prefix=strings[0]\n    for s in strings[1:]:\n        while not s.startswith(prefix): prefix=prefix[:-1]\n    return prefix",
            [_fc([[]],""), _fc([["flower","flow","flight"]],"fl"), _fc([["dog","racecar","car"]],""), _fc([["same"]],"same"), _fc([["","abc"]],""), _fc([["Привет","Призма"]],"При")], 8226),
    _coding("code-validate-ipv4", "valid_ipv4", "Define valid_ipv4(text). Accept exactly four decimal components 0..255; reject signs, whitespace, empty components, and leading zeroes except '0'.",
            "def valid_ipv4(text):\n    parts=text.split('.')\n    return len(parts)==4 and all(p.isascii() and p.isdigit() and (p=='0' or not p.startswith('0')) and 0<=int(p)<=255 for p in parts)",
            [_fc(["127.0.0.1"],True), _fc(["255.255.255.255"],True), _fc(["256.1.1.1"],False), _fc(["01.2.3.4"],False), _fc(["1.2.3"],False), _fc(["1.2.3.-1"],False)], 8227),
    _coding("code-parse-query", "parse_query", "Define parse_query(text). Split on &, then first =; missing = means empty value. Percent decoding is out of scope. Return dict key -> list of values in encounter order, including empty keys.",
            "def parse_query(text):\n    out={}\n    if text=='': return out\n    for part in text.split('&'):\n        key,sep,value=part.partition('='); out.setdefault(key,[]).append(value if sep else '')\n    return out",
            [_fc([""],{}), _fc(["a=1&b=2&a=3"],{'a':['1','3'],'b':['2']}), _fc(["flag"],{'flag':['']}), _fc(["x=a=b"],{'x':['a=b']}), _fc(["=v&&z="],{'':['v',''],'z':['']})], 8228),
    _coding("code-retry-delays", "retry_delays", "Define retry_delays(initial, factor, attempts, cap). Return attempts delays, multiplying for each next delay and capping each at cap. attempts <= 0 returns [].",
            "def retry_delays(initial,factor,attempts,cap):\n    out=[]; value=initial\n    for _ in range(max(0,attempts)):\n        out.append(min(value,cap)); value*=factor\n    return out",
            [_fc([1,2,5,10],[1,2,4,8,10]), _fc([3,3,4,20],[3,9,20,20]), _fc([1,2,0,9],[]), _fc([5,1,3,5],[5,5,5]), _fc([0.5,2,4,2],[0.5,1.0,2.0,2])], 8229),
]


def _simple_case(case_id: str, category: str, prompt: str, check: dict, seed: int, *, critical: bool = False, max_tokens: int = 512) -> dict:
    return {
        "case_id": case_id,
        "category": category,
        "critical": critical,
        "request": {
            "messages": [{"role": "system", "content": "Follow the requested output format exactly."}, {"role": "user", "content": prompt}],
            "generation": {"temperature": 0, "seed": seed, "max_tokens": max_tokens},
        },
        "check": check,
    }


_REASONING = [
    _simple_case("reasoning-trains", "reasoning_ru", "Поезд прошёл 180 км за 2 ч, затем 120 км за 1 ч. Верни только среднюю скорость всего пути в формате N км/ч.", {"type":"exact","expected":"100 км/ч","casefold":True}, 8301, critical=True),
    _simple_case("reasoning-workers", "reasoning_ru", "6 работников выполняют заказ за 15 дней при одинаковой производительности. За сколько дней его выполнят 10 работников? Верни только целое число.", {"type":"exact","expected":"9","casefold":False}, 8302),
    _simple_case("reasoning-discount-tax", "reasoning_ru", "Цена 2000 рублей. Сначала скидка 15%, затем налог 20% на цену после скидки. Верни только итог с двумя знаками: N.NN", {"type":"exact","expected":"2040.00","casefold":False}, 8303),
    _simple_case("reasoning-mixture", "reasoning_ru", "Смешали 2 л 10%-го раствора и 3 л 30%-го. Верни только итоговую концентрацию с символом процента.", {"type":"exact","expected":"22%","casefold":False}, 8304),
    _simple_case("reasoning-clock", "reasoning_ru", "Какой меньший угол между стрелками часов в 3:30? Верни только число градусов с символом °.", {"type":"exact","expected":"75°","casefold":False}, 8305),
    _simple_case("reasoning-sequence", "reasoning_ru", "Последовательность: 2, 6, 12, 20, 30. Следующий член строится по тому же правилу. Верни только число.", {"type":"exact","expected":"42","casefold":False}, 8306),
    _simple_case("reasoning-probability", "reasoning_ru", "В мешке 3 красных и 2 синих шара. Два шара достают без возвращения. Верни вероятность двух красных как несократимую дробь.", {"type":"exact","expected":"3/10","casefold":False}, 8307),
    _simple_case("reasoning-modular", "reasoning_ru", "Найди наименьшее неотрицательное число x: x ≡ 2 (mod 3) и x ≡ 3 (mod 5). Верни только x.", {"type":"exact","expected":"8","casefold":False}, 8308),
    _simple_case("reasoning-sets", "reasoning_ru", "В классе 30 учеников: 18 знают Python, 14 знают SQL, 7 знают оба. Сколько не знают ни Python, ни SQL? Верни только число.", {"type":"exact","expected":"5","casefold":False}, 8309),
    _simple_case("reasoning-rate", "reasoning_ru", "Труба A наполняет бак за 6 часов, B — за 3 часа. Они работают вместе. Верни только время заполнения в часах целым числом.", {"type":"exact","expected":"2","casefold":False}, 8310),
    _simple_case("reasoning-base-conversion", "reasoning_ru", "Переведи двоичное 101101 в десятичное. Верни только число.", {"type":"exact","expected":"45","casefold":False}, 8311),
]


_INSTRUCTION = [
    _simple_case("instruction-sorted-json", "instruction", "Верни только JSON-объект без markdown с ключами ascending и count для чисел 7, 2, 7, 1. ascending — отсортированный список с повторами.", {"type":"json_shape","schema":{"type":"object","required":["ascending","count"],"properties":{"ascending":{"type":"array"},"count":{"type":"integer"}},"additionalProperties":False},"expected":{"ascending":[1,2,7,7],"count":4}}, 8401, critical=True),
    _simple_case("instruction-three-lines", "instruction", "Верни ровно три строки: первая RED, вторая GREEN, третья BLUE. Никакого другого текста.", {"type":"text_constraints","required_regex":["^RED\\nGREEN\\nBLUE$"],"forbidden_regex":["```","[a-z]"],"max_lines":3}, 8402),
    _simple_case("instruction-csv-row", "instruction", "Верни ровно одну CSV-строку из значений alpha, 17, true без пробелов.", {"type":"exact","expected":"alpha,17,true","casefold":False}, 8403),
    _simple_case("instruction-nested-json", "instruction", "Верни только JSON: объект user с полями name='Ира' и active=true, плюс поле roles — список ['admin','reader'].", {"type":"json_shape","schema":{"type":"object","required":["user","roles"],"properties":{"user":{"type":"object"},"roles":{"type":"array"}},"additionalProperties":False},"expected":{"user":{"name":"Ира","active":True},"roles":["admin","reader"]}}, 8404),
    _simple_case("instruction-prefix-suffix", "instruction", "Верни одну строку, начинающуюся START:, затем ровно четыре цифры 2048, затем :END.", {"type":"text_constraints","required_regex":["^START:2048:END$"],"forbidden_regex":["```","\\n"," "],"max_lines":1}, 8405),
    _simple_case("instruction-unique-json", "instruction", "Верни только JSON-объект с единственным ключом unique, содержащим список первых появлений из ['b','a','b','c','a'].", {"type":"json_shape","schema":{"type":"object","required":["unique"],"properties":{"unique":{"type":"array"}},"additionalProperties":False},"expected":{"unique":["b","a","c"]}}, 8406),
    _simple_case("instruction-boolean-matrix", "instruction", "Верни только JSON-массив [[true,false],[false,true]].", {"type":"json_shape","schema":{"type":"array"},"expected":[[True,False],[False,True]]}, 8407),
    _simple_case("reasoning-order-constraints", "reasoning_ru", "Есть четыре задачи A, B, C, D. D выполняется раньше A; A раньше B; C выполняется сразу после B. Верни единственный порядок четырьмя буквами без разделителей.", {"type":"exact","expected":"DABC","casefold":False}, 8408),
    _simple_case("instruction-key-types", "instruction", "Верни только JSON-объект: id — целое 42, label — строка '42', enabled — false.", {"type":"json_shape","schema":{"type":"object","required":["id","label","enabled"],"properties":{"id":{"type":"integer"},"label":{"type":"string"},"enabled":{"type":"boolean"}},"additionalProperties":False},"expected":{"id":42,"label":"42","enabled":False}}, 8409),
]


_RETRIEVAL = [
    _simple_case("retrieval-config-precedence", "retrieval", "Используй только контекст. defaults: port=7000 mode=safe. region-eu: port=7100. service-orion: region=eu mode=fast. Каковы итоговые port и mode для orion? Верни port|mode.", {"type":"exact","expected":"7100|fast","casefold":False}, 8501, critical=True),
    _simple_case("retrieval-latest-event", "retrieval", "Используй только журнал: [09:00] api=up version=4; [09:05] db=down code=E7; [09:08] db=up code=OK; [09:10] api=up version=5. Верни последнее состояние db как state|code.", {"type":"exact","expected":"up|OK","casefold":False}, 8502),
    _simple_case("retrieval-table-join", "retrieval", "Таблица users: u1=Анна,u2=Борис,u3=Вера. Таблица jobs: j8 owner=u3 status=done; j9 owner=u1 status=wait. Для j8 верни имя владельца|статус.", {"type":"exact","expected":"Вера|done","casefold":False}, 8503),
    _simple_case("retrieval-decoy-version", "retrieval", "Документ A (устарел): nebula=1.4 checksum=OLD. Документ B (действует с 2026-08-01): nebula=2.1 checksum=7fa9. Вопрос относится к действующему документу. Верни version|checksum.", {"type":"exact","expected":"2.1|7fa9","casefold":False}, 8504),
    _simple_case("retrieval-multihop-owner", "retrieval", "Контекст: project p7 uses cluster c2. clusters: c1 owner=t1; c2 owner=t3. teams: t1=Atlas; t2=Boreal; t3=Cygnus. Верни команду-владельца project p7 одним словом.", {"type":"exact","expected":"Cygnus","casefold":False}, 8505),
]


def _legacy_cases() -> list[dict]:
    archive = "; ".join(
        f"archive-{n:03d} owner=ArchiveOwner{n:03d} port={6000+n} checksum={(n*7919)%100000:05d} status=retired"
        for n in range(41, 221)
    )
    nodes = [
        ("Ada","amber"),("Boris","blue"),("Cleo","coral"),("Daria","denim"),("Emil","emerald"),("Farah","fuchsia"),("Gleb","gold"),("Hana","hazel"),("Ilya","indigo"),("Jana","jade"),
        ("Kir","khaki"),("Lina","lilac"),("Marek","maroon"),("Nina","navy"),("Omar","olive"),("Petra","pearl"),("Quinn","quartz"),("Rita","red"),("Sven","silver"),("Tala","teal"),
        ("Umar","umber"),("Vera","violet"),("Wade","white"),("Xena","xanthic"),("Yara","yellow"),("Zane","zinc"),("Arun","apricot"),("Bela","beige"),("Cato","cyan"),("Dora","drab"),
        ("Eran","ecru"),("Faye","flax"),("Gina","green"),("Hugo","heliotrope"),("Ines","ivory"),("Juri","jet"),("Kira","kelly"),("Leon","lemon"),("Mira","magenta"),("Niko","ochre"),
    ]
    inventory = "; ".join(f"node-{i:03d} owner={owner} port={5100+i} color={color}" for i,(owner,color) in enumerate(nodes,1))
    cases = [
        {"case_id":"exact-fact-arithmetic","category":"instruction","critical":True,"request":{"messages":[{"role":"system","content":"Answer with only the requested value. Do not explain."},{"role":"user","content":"Give the capital of France, then a vertical bar, then the exact value of 12 multiplied by 12."}],"generation":{"temperature":0,"seed":8101,"max_tokens":32}},"check":{"type":"exact","expected":"Paris|144","casefold":True}},
        {"case_id":"prompt-json-schema","category":"instruction","critical":True,"request":{"messages":[{"role":"system","content":"Return one JSON object and no markdown. The exact keys are service, replicas, and healthy."},{"role":"user","content":"Report that the service named atlas has 3 replicas and is healthy."}],"generation":{"temperature":0,"seed":8102,"max_tokens":64}},"check":{"type":"json_shape","schema":{"type":"object","required":["service","replicas","healthy"],"properties":{"service":{"type":"string"},"replicas":{"type":"integer"},"healthy":{"type":"boolean"}},"additionalProperties":False},"expected":{"service":"atlas","replicas":3,"healthy":True}}},
        {"case_id":"code-unit-expression","category":"coding","critical":True,"request":{"messages":[{"role":"system","content":"Return only one Python lambda expression. Do not use markdown, imports, calls, attributes, or assignments."},{"role":"user","content":"Write a lambda with argument x that computes x squared plus two times x minus 3."}],"generation":{"temperature":0,"seed":8103,"max_tokens":64}},"check":{"type":"python_lambda_cases","argument":"x","cases":[{"input":-4,"expected":5},{"input":0,"expected":-3},{"input":3,"expected":12},{"input":11,"expected":140}]}},
        {"case_id":"instruction-constraints","category":"instruction","critical":False,"request":{"messages":[{"role":"system","content":"Follow every formatting constraint exactly."},{"role":"user","content":"Return exactly one line containing the three uppercase words ALPHA, BRAVO, CHARLIE in that order, separated by one comma and no spaces."}],"generation":{"temperature":0,"seed":8104,"max_tokens":32}},"check":{"type":"text_constraints","required_regex":["ALPHA,BRAVO,CHARLIE"],"forbidden_regex":["```","\\n","[a-z]"," "],"max_lines":1}},
        {"case_id":"tool-call-schema","category":"instruction","critical":True,"request":{"messages":[{"role":"system","content":"Use the required function and do not answer with prose."},{"role":"user","content":"Look up the service named cedar in the eu-west region."}],"generation":{"temperature":0,"seed":8105,"max_tokens":96},"tools":[{"type":"function","function":{"name":"lookup_service","description":"Look up one service in one region.","parameters":{"type":"object","required":["service","region"],"properties":{"service":{"type":"string"},"region":{"type":"string"}},"additionalProperties":False}}}],"tool_choice":{"type":"function","function":{"name":"lookup_service"}}},"check":{"type":"tool_call","function_name":"lookup_service","arguments":{"service":"cedar","region":"eu-west"}}},
        {"case_id":"long-context-needle","category":"retrieval","critical":True,"request":{"messages":[{"role":"system","content":"Use only the supplied inventory. Return exactly owner|port and no other text."},{"role":"user","content":"Archived records are irrelevant to the final inventory question but remain part of the supplied context. "+archive+"; Do not use an archived record as the answer."},{"role":"user","content":"Inventory snapshot follows. "+inventory+". Maintenance note: all odd-numbered nodes are checked on Monday and all even-numbered nodes on Thursday. The checksum labels are independent of owner names and ports. Ignore colors and maintenance schedules. Which owner and port belong to node-037?"}],"generation":{"temperature":0,"seed":8106,"max_tokens":32}},"check":{"type":"exact","expected":"Kira|5137","casefold":False}},
    ]
    return cases


def build_suite() -> dict:
    """Return a defensive copy of the pre-execution fixture catalog."""
    cases = _legacy_cases() + _CODING + _REASONING + _INSTRUCTION + _RETRIEVAL
    return copy.deepcopy({"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "cases": cases})


SUITE = build_suite()
