with Ada.Assertions; use Ada.Assertions;

with Show_Ineffective_Statements; use Show_Ineffective_Statements;

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

   procedure Test_Swap2 is
      X1, Y1 : T := 1;
      X2     : T := 2;
      Y2     : T := 3;
      X3     : T := T'First;
      Y3     : T := T'Last;
   begin
      Swap2 (X1, Y1);
      pragma Assert (X1 = 1 and Y1 = 1, "Should have no effect");
      Swap2 (X2, Y2);
      pragma Assert (X2 = 3 and Y2 = 2, "Simple swap");
      Swap2 (X3, Y3);
      pragma Assert (X3 = T'Last and Y3 = T'First, "Extreme values");
   end Test_Swap2;

begin
   Check_Assertions_Enabled;
   Test_Swap2;
end Tests;
