# class Solution:
#     def twoSum(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: List[int]
#         """
#         hashmap = {}
#         for index, num in enumerate(nums):
#             another_num = target - num
#             if another_num in hashmap:
#                 return [hashmap[another_num], index]
#             hashmap[num] = index
#         return None
#
#
# # Definition for singly-linked list.
# # class ListNode:
# #     def __init__(self, x):
# #         self.val = x
# #         self.next = None
#
# class Solution:
#     def addTwoNumbers(self, l1, l2):
#         """
#         :type l1: ListNode
#         :type l2: ListNode
#         :rtype: ListNode
#         """
#         re = ListNode(0)
#         r = re
#         carry = 0
#         while (l1 or l2):
#             x = 0
#             y = 0
#             if l1:
#                 x = l1.val
#                 l1 = l1.next
#             if l2:
#                 y = l2.val
#                 l2 = l2.next
#             carry += (x + y)
#             s = carry
#             carry = (s >= 10)
#             if carry:
#                 r.next = ListNode(s - 10)
#             else:
#                 r.next = ListNode(s)
#             r = r.next
#         if carry:
#             r.next = ListNode(1)
#         return re.next

s = " "
bignum = 0
hashmap = {}
for index, char in enumerate(s):
    cumunum = 1
    hashmap = {}
    hashmap[char] = cumunum
    while (index + cumunum) < len(s):
        if s[index + cumunum] in hashmap:
            break
        hashmap[s[index + cumunum]] = cumunum
        cumunum += 1
    if bignum < cumunum:
        bignum = cumunum

print(bignum)