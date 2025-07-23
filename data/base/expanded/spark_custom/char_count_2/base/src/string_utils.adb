package body String_Utils is

   function Count (Str : String; Char : Character) return Natural is
      Result : Natural := 0;
   begin
      for I in Str'Range loop
         if Str (I) = Char then
            Result := Result + 1;
         end if;
      end loop;
      return Result;
   end Count;

end String_Utils;
