package Array_Utils is

   type Nat_Array is array (Integer range <>) of Natural;

   procedure Update (A : in out Nat_Array; I, J, P, Q : Integer)
   with
     Pre =>
       ((if I >= 0 and J >= 0
         then I <= Integer'Last - J
         elsif I < 0 and J < 0
         then I >= Integer'First - J
         else True)  -- Check that I + J does not overflow
        and then I + J in A'Range  -- Check that I + J is an index in A
        and then Q /= 0  -- So there can't be a division by zero
        and then not (P = Integer'First
                      and Q = -1)  -- As abs Integer'First is out of range
        and then P / Q >= 1);  -- Ensure P / Q is a natural number

end Array_Utils;
