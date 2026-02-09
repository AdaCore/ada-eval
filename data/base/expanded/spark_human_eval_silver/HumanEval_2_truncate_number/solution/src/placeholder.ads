package Placeholder is

   function Truncate_Number (Number : Float) return Float
   is (Number - Float'Floor (Number))
   with
     Pre  => Number >= 0.0,
     Post =>
       Truncate_Number'Result >= 0.0
       and Truncate_Number'Result < 1.0
       and Number = Float'Floor (Number) + Truncate_Number'Result;
   --  Given a positive floating point number, it can be decomposed into and
   --  integer part (largest integer smaller than given number) and decimals
   --  (leftover part always smaller than 1).
   --  Return the decimal part of the number.
   --  >>> Truncate_Number (3.5)
   --  0.5

end Placeholder;
