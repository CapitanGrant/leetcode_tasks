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

    def search(self, nums: List[int], target: int) -> int:
        low = 0
        high = len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            guess = nums[mid]
            if guess == target:
                return mid
            if guess > target:
                high = mid - 1
            else:
                low = mid + 1
        return -1

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

    def isValid(self, s: str) -> bool:
        pass

    def mySqrt(self, x: int) -> int:
        low = 0
        high = x
        while low <= high:
            mid = (low + high) // 2
            if mid ** 2 <= x <= (mid + 1) ** 2:
                return mid
            if mid ** 2 > x:
                high = mid - 1
            else:
                low = mid + 1
        return -1

    def missingNumber(self, nums: List[int]) -> int:
        sum_num = sum(nums)
        len_num = len(nums)
        res_num = sum([i for i in range(0, len_num+1)])
        return res_num - sum_num

    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1 = set(nums1)
        nums2 = set(nums2)
        lst = list(nums1 & nums2)
        return lst

    def isBadVersion(self, version: int, bad: int) -> bool:
        if version == bad:
            return True
        return False

    def firstBadVersion(self, n: int) -> int:
        low = 0
        high = n - 1
        while low < high:
            mid = (low + high) // 2
            if self.sBadVersion(mid):
                high = mid
            else:
                low = mid + 1
        return low

sol = Solution()
print(sol.firstBadVersion(5))

# print(sol.mySqrt(9))

# print(sol.isValid(s="()")) # true
# print(sol.isValid(s="()[]{}")) # true
# print(sol.isValid(s="(]")) # false
#
#
# print(sol.longestCommonPrefix(strs=["flower", "flow", "flight"]))
# print(sol.longestCommonPrefix(strs=["dog","racecar","car"]))
#
#
# print(sol.addBinary('000100', '110010'))  # 110110
# print(sol.addBinary('1010', '1011'))  # 10101
