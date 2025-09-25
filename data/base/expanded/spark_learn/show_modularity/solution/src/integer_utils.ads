package Integer_Utils is

   procedure Increment (X : in out Integer)
   with Pre => X < Integer'Last, Post => X = X'Old + 1;

   procedure Increment_Twice (X : in out Integer)
   with Pre => X < Integer'Last - 1, Post => X = X'Old + 2;

end Integer_Utils;
