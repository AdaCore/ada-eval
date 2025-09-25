with Ada.Assertions; use Ada.Assertions;
with Ada.Numerics.Discrete_Random;

with Integer_Utils; use Integer_Utils;

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

   procedure Test_Increment is
      X : Integer := 0;
   begin
      Increment (X);
      pragma Assert (X = 1);
      Increment_Twice (X);
      pragma Assert (X = 3);
      X := Integer'Last - 1;
      Increment (X);
      pragma Assert (X = Integer'Last);
      X := Integer'First;
      Increment_Twice (X);
      pragma Assert (X = Integer'First + 2);
   end Test_Increment;

begin
   Check_Assertions_Enabled;
   Test_Increment;
end Tests;
