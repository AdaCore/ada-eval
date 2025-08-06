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

   function Contains (A : Array_Of_Positives; E : Positive) return Boolean is
      Result : Integer;
   begin
      Search_Array (A, E, Result);
      pragma Unreferenced (Result);
      return True;
   exception
      when Not_Found =>
         return False;
   end Contains;

end Search_Array;
