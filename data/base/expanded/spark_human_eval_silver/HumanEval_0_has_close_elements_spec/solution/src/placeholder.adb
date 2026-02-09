package body Placeholder is

   function Has_Close_Elements
     (Numbers : Float_Array; Threshold : Float) return Boolean
   is
      X, Y : Float;
   begin
      --  If threshold is non-positive, no two values can be closer than 0
      if Threshold <= 0.0 then
         return False;
      end if;

      --  If there are less than 2 elements, cannot have close elements
      if Numbers'Length < 2 then
         return False;
      end if;

      for I in Numbers'First .. Numbers'Last - 1 loop
         for J in I + 1 .. Numbers'Last loop
            --   Make X the larger of the two values, and Y the smaller
            if (Numbers (I) = Numbers (J)) then
               --  Identical values have distance 0,
               --  which is less than any positive threshold
               return True;
            elsif (Numbers (I) > Numbers (J)) then
               X := Numbers (I);
               Y := Numbers (J);
            else
               X := Numbers (J);
               Y := Numbers (I);
            end if;

            --  Check if |X - Y| < Threshold
            --  When both values have the same sign, subtraction is safe
            --  (cannot overflow/underflow)
            if (X >= 0.0 and Y >= 0.0) or (X < 0.0 and Y < 0.0) then
               if X - Y < Threshold then
                  return True;
               end if;
            else
               --  Different signs - use addition-based comparison
               --  Check if X - Y < Threshold, i.e., X < Y + Threshold
               if Y >= 0.0 and then Threshold > Float'Last - Y then
                  --  Y + Threshold would overflow, so difference < threshold
                  return True;
               elsif X < Y + Threshold then
                  return True;
               end if;
            end if;
         end loop;
      end loop;
      return False;
   end Has_Close_Elements;

end Placeholder;
