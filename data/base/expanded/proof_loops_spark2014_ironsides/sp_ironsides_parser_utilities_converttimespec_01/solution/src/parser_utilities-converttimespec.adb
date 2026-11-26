----------------------------------------------------------------
--  IRONSIDES - DNS SERVER
--
--  By: Martin C. Carlisle and Barry S. Fagin
--     Department of Computer Science
--     United States Air Force Academy
--
--  Modified by: Altran UK Limited
--
--  This is free software; you can redistribute it and/or
--  modify without restriction.  We do ask that you please keep
--  the original author information, and clearly indicate if the
--  software has been modified.
--
--  This software is distributed in the hope that it will be useful,
--  but WITHOUT ANY WARRANTY; without even the implied warranty
--  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
----------------------------------------------------------------

separate (Parser_Utilities)
procedure ConvertTimeSpec
  (S       : in RR_Type.LineFromFileType;
   BegIdx  : in RR_Type.LineLengthIndex;
   EndIdx  : in RR_Type.LineLengthIndex;
   RetVal  : out Unsigned_Types.Unsigned32;
   Success : in out Boolean)
is
   Tmp         : Long_Long_Integer := 0;
   Blob        : Long_Long_Integer := 0;
   CurrentChar : Character;

   function MultValue (Char : in Character) return Natural
   with
     Pre  => IsMult (Char),
     Post => MultValue'Result >= 0 and MultValue'Result <= 60 * 60 * 24 * 7;

   function MultValue (Char : in Character) return Natural is
      C      : Character;
      RetVal : Natural;
   begin
      C := Ada.Characters.Handling.To_Upper (Char);
      case C is
         when 'S'    =>
            RetVal := 1;

         when 'M'    =>
            RetVal := SecondsInAMinute;

         when 'H'    =>
            RetVal := SecondsInAMinute * MinutesInAnHour;

         when 'D'    =>
            RetVal := (SecondsInAMinute * MinutesInAnHour) * HoursInADay;

         when 'W'    =>
            RetVal :=
              ((SecondsInAMinute * MinutesInAnHour) * HoursInADay)
              * DaysInAWeek;

         when others =>
            RetVal := 0;   -- will never happen if precondition is met
      end case;
      return RetVal;
   end MultValue;

begin
   for I in Integer range BegIdx .. EndIdx loop
      pragma
        Loop_Invariant
          (Blob >= 0
           and Blob <= RR_Type.SOA_Record_Type.Max_Time_Val
           and Tmp >= 0
           and Tmp <= RR_Type.SOA_Record_Type.Max_Time_Val);
      CurrentChar := S (I);
      if CurrentChar >= '0' and CurrentChar <= '9' then
         Blob :=
           10 * Blob + (Character'Pos (CurrentChar) - Character'Pos ('0'));
         if Blob > RR_Type.SOA_Record_Type.Max_Time_Val then
            Success := False;
         end if;
      elsif IsMult (CurrentChar) then
         if Blob = 0 then
            Success := False;
         else
            Blob := Blob * Long_Long_Integer (MultValue (CurrentChar));
            Tmp := Tmp + Blob;
            if Blob > RR_Type.SOA_Record_Type.Max_Time_Val
              or Tmp > RR_Type.SOA_Record_Type.Max_Time_Val
            then
               Success := False;
            end if;
            Blob := 0;
         end if;
      else
         Success := False;
      end if;
      exit when Success = False;
   end loop;
   if not Success then
      Tmp := 0;
   elsif Tmp = 0 then
      --  handle case when time specifier is just a number
      --(no mult)
      Tmp := Blob;
   end if;
   RetVal := Unsigned_Types.Unsigned32 (Tmp);
end ConvertTimeSpec;
