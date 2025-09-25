package Integer_Utils is

   procedure Absolute (X : in out Integer)
   with Post => X >= 0;

end Integer_Utils;
