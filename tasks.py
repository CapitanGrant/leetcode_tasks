from typing import List


class Solution:
    def addBinary(self, a: str, b: str) -> str:
        num = 0
        lst = []
        max_len = max(len(a), len(b))
        a = a.zfill(max_len)
        b = b.zfill(max_len)
        for i in range(max_len - 1, -1, -1):
            sum = num
            sum += 1 if a[i] == '1' else 0
            sum += 1 if b[i] == '1' else 0
            lst.insert(0, '1' if sum % 2 else '0')
            num = 1 if sum > 1 else 0
        if num:
            lst.insert(0, '1')

        return str(''.join(lst))

    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""

        min_len = len(min(strs, key=len))
        lcp = ""

        for i in range(min_len):
            char = strs[0][i]
            for j in strs:
                if i >= len(j) or j[i] != char:
                    return lcp
            lcp += char

        return lcp

sol = Solution()

print(sol.longestCommonPrefix(strs=["flower", "flow", "flight"]))
print(sol.longestCommonPrefix(strs=["dog","racecar","car"]))


# print(sol.addBinary('000100', '110010'))  # 110110
# print(sol.addBinary('1010', '1011'))  # 10101
