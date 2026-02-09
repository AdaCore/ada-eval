with Ada.Assertions; use Ada.Assertions;

with Types;       use Types;
with Placeholder; use Placeholder;

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

   function X (Numbers : Float_Array; Threshold : Float) return Boolean
   renames Placeholder.Has_Close_Elements;

begin
   Check_Assertions_Enabled;

   --  Basic tests
   pragma Assert (X ([1.0], 0.0) = False);
   pragma Assert (X ([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) = True);
   pragma Assert (X ([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) = False);
   pragma Assert (X ([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) = True);
   pragma Assert (X ([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) = False);
   pragma Assert (X ([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) = True);
   pragma Assert (X ([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) = True);
   pragma Assert (X ([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) = False);
   pragma Assert (X ([1.0, 1.0], 0.0) = False);
   pragma Assert (X ([1.0, 1.0], Float'Model_Small) = True);
   pragma Assert (X ([Float'Last, Float'First], Float'Model_Small) = False);
   pragma
     Assert (X ([0.0, 2.0 * Float'Model_Small], Float'Model_Small) = False);
   pragma
     Assert (X ([0.0, Float'Model_Small], 2.0 * Float'Model_Small) = True);

   --  Tests that may cause overflow in loop implementation
   pragma
     Assert
       (X
          ((Positive'Last - 2 => 0.0,
            Positive'Last - 1 => 1.0,
            Positive'Last     => 1.1),
           0.05)
          = False);
   pragma
     Assert
       (X
          ((Positive'Last - 2 => 0.0,
            Positive'Last - 1 => 1.0,
            Positive'Last     => 1.1),
           0.2)
          = True);

   -- Tests that might break naive abs(X - Y) < Threshold implementation

   -- Overflow case: X - Y would overflow (large positive - large negative)
   pragma Assert (X ([Float'Last, -Float'Last / 2.0], 1.0) = False);
   pragma Assert (X ([Float'Last / 2.0, -Float'Last / 2.0], 1.0) = False);

   -- Underflow case: X - Y would underflow (large negative - large positive)
   pragma Assert (X ([Float'First, Float'Last / 2.0], 1.0) = False);
   pragma Assert (X ([Float'First / 2.0, Float'Last], 1.0) = False);

   -- Both at Float'Last - should be considered close with positive threshold
   pragma Assert (X ([Float'Last, Float'Last], 1.0) = True);
   pragma Assert (X ([Float'Last, Float'Last], 0.1) = True);
   pragma Assert (X ([Float'Last, Float'Last], Float'Model_Small) = True);

   -- Both at Float'First - should be considered close with positive threshold
   pragma Assert (X ([Float'First, Float'First], 1.0) = True);
   pragma Assert (X ([Float'First, Float'First], 0.1) = True);

   -- Near Float'Last with actual consecutive values (Y + Threshold might overflow)
   pragma Assert (X ([Float'Last, Float'Pred (Float'Last)], 1.0E32) = True);
   pragma Assert (X ([Float'Last, Float'Pred (Float'Last)], 1.0E31) = False);

   -- Near Float'First with actual consecutive values
   pragma Assert (X ([Float'First, Float'Succ (Float'First)], 1.0E32) = True);
   pragma Assert (X ([Float'First, Float'Succ (Float'First)], 1.0E31) = False);

   -- Extreme thresholds
   pragma Assert (X ([Float'Last, Float'First], Float'First) = False);
   pragma Assert (X ([Float'Last, Float'First], Float'Last) = False);
   pragma Assert (X ([0.0, 0.0], Float'First) = False);
   pragma Assert (X ([-1.0, Float'Last], Float'Last) = False);

   -- Negative threshold should always return False
   pragma Assert (X ([1.0, 1.0], -1.0) = False);
   pragma Assert (X ([1.0, 1.5], -0.1) = False);

   -- Tests with consecutive floating point values at extremes
   -- The spacing at Float'Last is ~2E+31, so consecutive values are very far apart
   pragma Assert (X ([Float'Last, Float'Pred (Float'Last)], 1.0E20) = False);
   pragma Assert (X ([Float'Last, Float'Pred (Float'Last)], 1.0E31) = False);
   pragma Assert (X ([Float'Last, Float'Pred (Float'Last)], 1.0E32) = True);

   -- Same for Float'First
   pragma Assert (X ([Float'First, Float'Succ (Float'First)], 1.0E20) = False);
   pragma Assert (X ([Float'First, Float'Succ (Float'First)], 1.0E31) = False);
   pragma Assert (X ([Float'First, Float'Succ (Float'First)], 1.0E32) = True);

   -- Values near Float'First
   pragma
     Assert
       (X
          ([Float'First,
            Float'Succ (Float'First),
            Float'Succ (Float'Succ (Float'First))],
           1.0E31)
          = False);
   pragma
     Assert
       (X
          ([Float'First,
            Float'Succ (Float'First),
            Float'Succ (Float'Succ (Float'First))],
           3.0E31)
          = True);

   -- Values near Float'Last
   pragma
     Assert
       (X
          ([Float'Last,
            Float'Pred (Float'Last),
            Float'Pred (Float'Pred (Float'Last))],
           1.0E31)
          = False);
   pragma
     Assert
       (X
          ([Float'Last,
            Float'Pred (Float'Last),
            Float'Pred (Float'Pred (Float'Last))],
           3.0E31)
          = True);

end Tests;
