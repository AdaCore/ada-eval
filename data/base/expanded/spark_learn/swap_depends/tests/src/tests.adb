with Ada.Assertions; use Ada.Assertions;

with Show_Swap; use Show_Swap;

procedure Tests is
   procedure Check_Assertions_Enabled is
   begin
      begin
         pragma Assert (False, "Should raise");
      exception
         when others =>
            return; -- properly raised
      end;
      raise Program_Error with "Assertions not enabled";
   end Check_Assertions_Enabled;

   procedure Test_Swap is
      X1, Y1 : Positive := 1;
      X2     : Positive := 2;
      Y2     : Positive := 3;
      X3     : Positive := 1;
      Y3     : Positive := Positive'Last;
   begin
      Swap (X1, Y1);
      pragma Assert (X1 = 1 and Y1 = 1, "Should have no effect");
      Swap (X2, Y2);
      pragma Assert (X2 = 3 and Y2 = 2, "Simple swap");
      Swap (X3, Y3);
      pragma Assert (X3 = Positive'Last and Y3 = 1, "Extreme values");
   end Test_Swap;

   procedure Test_Identity is
      X1, Y1 : Positive := 1;
      X2     : Positive := 2;
      Y2     : Positive := 3;
      X3     : Positive := 1;
      Y3     : Positive := Positive'Last;
   begin
      Identity (X1, Y1);
      pragma Assert (X1 = 1 and Y1 = 1, "Same values");
      Identity (X2, Y2);
      pragma Assert (X2 = 2 and Y2 = 3, "Simple identity");
      Identity (X3, Y3);
      pragma Assert (X3 = 1 and Y3 = Positive'Last, "Extreme values");
   end Test_Identity;

begin
   Check_Assertions_Enabled;
   Test_Swap;
   Test_Identity;
end Tests;
