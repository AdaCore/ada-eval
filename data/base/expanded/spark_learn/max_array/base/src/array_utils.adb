package body Array_Utils is

   function Max_Array (A : Array_Of_Naturals) return Natural is
      Max : Natural := Natural'First;
   begin
      for I in A'Range loop
         if A (I) > Max then
            Max := A (I);
         end if;
      end loop;
      return Max;
   end Max_Array;

end Array_Utils;
