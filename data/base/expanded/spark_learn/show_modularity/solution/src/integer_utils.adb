package body Integer_Utils is

   procedure Increment (X : in out Integer) is
   begin
      X := X + 1;
   end Increment;

   procedure Increment_Twice (X : in out Integer) is
   begin
      Increment (X);
      Increment (X);
   end Increment_Twice;

end Integer_Utils;
