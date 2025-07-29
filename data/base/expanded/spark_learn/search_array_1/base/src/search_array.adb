package body Search_Array is

   procedure Search_Array
     (A : Array_Of_Positives; E : Positive; Result : out Integer) is
   begin
      for I in A'Range loop
         if A (I) = E then
            Result := I;
            return;
         end if;
      end loop;
      raise Not_Found;
   end Search_Array;

end Search_Array;
