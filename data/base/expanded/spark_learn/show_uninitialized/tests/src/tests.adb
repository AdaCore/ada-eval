with Ada.Assertions; use Ada.Assertions;
with Ada.Numerics.Discrete_Random;

with Array_Utils; use Array_Utils;

procedure Tests is

   procedure Shuffle_Array (Arr : in out Array_Utils.Array_Of_Naturals) is
      subtype Array_Index is Integer range Arr'Range;
      package Random_Index is new Ada.Numerics.Discrete_Random (Array_Index);
      use Random_Index;

      G          : Generator;
      Temp       : Natural;
      Random_Idx : Integer;
   begin
      Reset (G);

      -- Fisher-Yates shuffle
      for I in reverse Arr'Range loop
         Random_Idx := Random (G, Arr'First, I);
         Temp := Arr (I);
         Arr (I) := Arr (Random_Idx);
         Arr (Random_Idx) := Temp;
      end loop;
   end Shuffle_Array;

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

   procedure Test_Max_Array is
      Arr1       : constant Array_Of_Naturals := [1];
      Arr2       : constant Array_Of_Naturals :=
        [Natural'First, Natural'First + 1];
      Arr3       : constant Array_Of_Naturals := [Natural'Last, Natural'First];
      Arr4       : constant Array_Of_Naturals :=
        [Natural'Last, 0, Natural'First];
      Arr5       : Array_Of_Naturals (1 .. 10_000) :=
        (for I in 1 .. 10_000 => I);
      Arr6_First : constant Integer := Integer'Last - 9999;
      Arr6       : Array_Of_Naturals (Arr6_First .. Integer'Last) :=
        (for I in Arr6_First .. Integer'Last => I);
      Res1       : constant Natural := Max_Array (Arr1);
      Res2       : constant Natural := Max_Array (Arr2);
      Res3       : constant Natural := Max_Array (Arr3);
      Res4       : constant Natural := Max_Array (Arr4);
      Res5       : Natural;
      Res6       : Natural;
   begin
      Shuffle_Array (Arr5);
      Res5 := Max_Array (Arr5);
      Shuffle_Array (Arr6);
      Res6 := Max_Array (Arr6);
      pragma Assert (Res1 = 1, "Single item");
      pragma Assert (Res2 = Natural'First + 1, "Two small items");
      pragma Assert (Res3 = Natural'Last, "Extreme values");
      pragma Assert (Res4 = Natural'Last, "Three items");
      pragma Assert (Res5 = 10_000, "Large array");
      pragma Assert (Res6 = Integer'Last, "Large array");
   end Test_Max_Array;

begin
   Check_Assertions_Enabled;
   Test_Max_Array;
end Tests;
