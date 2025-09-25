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

   procedure Test_Absolute is
      X : Integer := 0;
   begin
      Absolute (X);
      pragma Assert (X = 0);
      X := -3;
      Absolute (X);
      pragma Assert (X = 3);
      Absolute (X);
      pragma Assert (X = 3);
      X := Integer'First + 1;
      Absolute (X);
      pragma Assert (X = Integer'Last);
      Absolute (X);
      pragma Assert (X = Integer'Last);
   end Test_Absolute;

begin
   Check_Assertions_Enabled;
   Test_Absolute;
end Tests;
