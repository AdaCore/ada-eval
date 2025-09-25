package Integer_Utils is

   procedure Increment (X : in out Integer)
   with Pre => X < Integer'Last;

   procedure Increment_Twice (X : in out Integer)
   with Pre => X < Integer'Last - 1;

end Integer_Utils;
