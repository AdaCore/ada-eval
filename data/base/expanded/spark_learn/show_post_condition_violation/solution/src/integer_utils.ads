package Integer_Utils is

   procedure Absolute (X : in out Integer)
   with Pre => X /= Integer'First, Post => X >= 0;

end Integer_Utils;
