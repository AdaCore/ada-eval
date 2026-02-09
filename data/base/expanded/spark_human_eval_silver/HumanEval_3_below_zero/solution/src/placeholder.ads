with Types; use Types;

package Placeholder is

   function Below_Zero (Operations : Integer_Array) return Boolean;
   --  Given a array of deposit and withdrawal operations on a bank account, in
   --  the order they were performed. Assuming the account balance starts at
   --  zero, this function will detect if at any point the balance of account
   --  falls below zero. If it does, it will return True. Otherwise it will
   --  return False.
   --  >>> Below_Zero ([1, 2, 3])
   --  False
   --  >>> Below_Zero ([1, 2, -4, 5])
   --  True

end Placeholder;
