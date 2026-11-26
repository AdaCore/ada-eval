separate (Memcp.Extractor)
function Valid_Utf8 (S : String) return Boolean is
   I : Integer := S'First;
   --  Index of the octet that starts the sequence under examination.

   function Byte (K : Integer) return Natural
   is (Character'Pos (S (K)))
   with Pre => K in S'Range;
   --  The octet at K as a number.

   function Cont (K : Integer) return Boolean
   is (K in S'Range and then Byte (K) in 16#80# .. 16#BF#);
   --  True when K is in range and holds a 10xxxxxx continuation octet.
begin
   while I <= S'Last loop
      pragma Loop_Variant (Increases => I);
      declare
         B0 : constant Natural := Byte (I);
      begin
         if B0 <= 16#7F# then
            --  ASCII
            I := I + 1;
         elsif B0 in 16#C2# .. 16#DF# then
            --  2-byte
            if not Cont (I + 1) then
               return False;
            end if;
            I := I + 2;
         elsif B0 = 16#E0# then
            --  3-byte, no overlong
            if I + 2 > S'Last
              or else Byte (I + 1) not in 16#A0# .. 16#BF#
              or else not Cont (I + 2)
            then
               return False;
            end if;
            I := I + 3;
         elsif B0 = 16#ED# then
            --  3-byte, no surrogate
            if I + 2 > S'Last
              or else Byte (I + 1) not in 16#80# .. 16#9F#
              or else not Cont (I + 2)
            then
               return False;
            end if;
            I := I + 3;
         elsif B0 in 16#E1# .. 16#EC# | 16#EE# .. 16#EF# then
            --  3-byte
            if not Cont (I + 1) or else not Cont (I + 2) then
               return False;
            end if;
            I := I + 3;
         elsif B0 = 16#F0# then
            --  4-byte, no overlong
            if I + 3 > S'Last
              or else Byte (I + 1) not in 16#90# .. 16#BF#
              or else not Cont (I + 2)
              or else not Cont (I + 3)
            then
               return False;
            end if;
            I := I + 4;
         elsif B0 = 16#F4# then
            --  4-byte, <= U+10FFFF
            if I + 3 > S'Last
              or else Byte (I + 1) not in 16#80# .. 16#8F#
              or else not Cont (I + 2)
              or else not Cont (I + 3)
            then
               return False;
            end if;
            I := I + 4;
         elsif B0 in 16#F1# .. 16#F3# then
            --  4-byte
            if not Cont (I + 1)
              or else not Cont (I + 2)
              or else not Cont (I + 3)
            then
               return False;
            end if;
            I := I + 4;
         else
            --  C0,C1,F5..FF, stray cont
            return False;
         end if;
      end;
   end loop;
   return True;
end Valid_Utf8;
