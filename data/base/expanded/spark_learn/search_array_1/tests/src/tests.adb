with Ada.Assertions; use Ada.Assertions;

with Search_Array; use Search_Array;

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

   procedure Test_Search_Array is
      -- Test arrays
      Large_Array         : Array_Of_Positives := (for I in 1 .. 1000 => I);
      Unusual_Range_Array : constant Array_Of_Positives (10 .. 12) :=
        [100, 200, 300];

      Result : Integer;
   begin

      -- Find first element
      Search_Array.Search_Array
        (Large_Array, Large_Array (Large_Array'First), Result);
      pragma
        Assert
          (Result = Large_Array'First, "Should find element at the beginning");

      -- Find last element
      Search_Array.Search_Array
        (Large_Array, Large_Array (Large_Array'Last), Result);
      pragma
        Assert (Result = Large_Array'Last, "Should find element at the end");

      -- Find middle element
      Search_Array.Search_Array
        (Unusual_Range_Array,
         Unusual_Range_Array (Unusual_Range_Array'First + 1),
         Result);
      pragma
        Assert
          (Result = Unusual_Range_Array'First + 1,
           "Should find element in the middle");

      -- Find non-existing element
      begin
         Search_Array.Search_Array (Large_Array, 1001, Result);
      exception
         when Not_Found =>
            return; -- expected exception
      end;
      raise Program_Error with "exception Not_Found not raised as expected";
   end Test_Search_Array;

begin
   Check_Assertions_Enabled;
   Test_Search_Array;
end Tests;
