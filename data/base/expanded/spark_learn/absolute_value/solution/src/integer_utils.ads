package Integer_Utils is

   procedure Absolute_Value (X : Integer; R : out Natural)
   with Pre => X /= Integer'First;

end Integer_Utils;
