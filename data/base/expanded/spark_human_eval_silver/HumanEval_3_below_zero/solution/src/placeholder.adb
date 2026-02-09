package body Placeholder is

   function Below_Zero (Operations : Integer_Array) return Boolean is
      Balance        : Integer := 0;
      Overflow_Count : Natural := 0;
      T              : Integer;
   begin
      for I in Operations'Range loop
         T := Operations (I);

         pragma Loop_Invariant (Overflow_Count <= I - Operations'First);
         pragma Loop_Invariant (Overflow_Count > 0 or Balance >= 0);

         if T > 0 and then Balance > Integer'Last - T then
            --  Would overflow: increment counter and compute wrapped balance
            --  Wrapped = Balance + T - 2^32
            --          = (Balance + Integer'First) + (T - Integer'Last - 1)
            Overflow_Count := Overflow_Count + 1;
            Balance := (Balance + Integer'First) + (T - Integer'Last - 1);

         elsif T < 0 and then Balance < Integer'First - T then
            --  Would underflow: decrement counter and compute wrapped balance
            --  Wrapped = Balance + T + 2^32
            --          = (Balance - Integer'First) + (T + Integer'Last + 1)
            Overflow_Count := Overflow_Count - 1;
            Balance := (Balance - Integer'First) + (T + Integer'Last + 1);

         else
            --  Normal case: no overflow or underflow
            Balance := Balance + T;
         end if;

         --  True balance is < 0 only when Overflow_Count = 0 and Balance < 0
         if Overflow_Count = 0 and then Balance < 0 then
            return True;
         end if;
      end loop;

      return False;
   end Below_Zero;

end Placeholder;
