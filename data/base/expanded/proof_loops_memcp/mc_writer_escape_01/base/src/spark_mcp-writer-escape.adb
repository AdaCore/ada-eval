separate (Spark_Mcp.Writer)
function Escape (S : String) return String is
   Buf  : String (1 .. Max_Expansion * S'Length) := (others => ' ');
   --  The escaped content, built in place. Sized for the worst case, so one
   --  pass never overflows; initialized in full so flow analysis sees
   --  Buf (1 .. Last) as defined, and the tail past Last is unused.

   Last : Natural := 0;
   --  Index in Buf of the last character written.
begin
   for I in S'Range loop

      declare
         C    : constant Character := S (I);
         Code : constant Natural   := Character'Pos (C);
      begin
         case C is
            when '"' =>
               Buf (Last + 1 .. Last + 2) := "\""";
               Last := Last + 2;
            when '\' =>
               Buf (Last + 1 .. Last + 2) := "\\";
               Last := Last + 2;
            when Character'Val (8) =>   --  backspace
               Buf (Last + 1 .. Last + 2) := "\b";
               Last := Last + 2;
            when Character'Val (9) =>   --  tab
               Buf (Last + 1 .. Last + 2) := "\t";
               Last := Last + 2;
            when Character'Val (10) =>  --  line feed
               Buf (Last + 1 .. Last + 2) := "\n";
               Last := Last + 2;
            when Character'Val (12) =>  --  form feed
               Buf (Last + 1 .. Last + 2) := "\f";
               Last := Last + 2;
            when Character'Val (13) =>  --  carriage return
               Buf (Last + 1 .. Last + 2) := "\r";
               Last := Last + 2;
            when Character'Val (0)  .. Character'Val (7)
               |  Character'Val (11)
               |  Character'Val (14) .. Character'Val (31) =>
               --  Any other control character: \u00XX.
               Buf (Last + 1 .. Last + 6) :=
                 "\u00" & Hex_Digit (Code / 16) & Hex_Digit (Code mod 16);
               Last := Last + 6;
            when others =>
               Buf (Last + 1) := C;
               Last := Last + 1;
         end case;
      end;
   end loop;

   return Buf (1 .. Last);
end Escape;
