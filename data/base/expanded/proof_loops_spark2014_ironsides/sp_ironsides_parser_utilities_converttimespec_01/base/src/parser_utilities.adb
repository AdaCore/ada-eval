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

--  Extracted from spark2014, testsuite/gnatprove/tests/ironsides.
--  ConvertTimeSpec, its nested MultValue and the body-local IsMult it calls
--  are copied verbatim; everything else in the 1300-line upstream body is
--  dropped, as are the constants only the dropped subprograms used.

with Ada.Characters.Handling, RR_Type.SOA_Record_Type;

package body Parser_Utilities
  with SPARK_Mode
is

   SecondsInAMinute : constant Natural := 60;
   MinutesInAnHour  : constant Natural := 60;
   HoursInADay      : constant Natural := 24;
   DaysInAWeek      : constant Natural := 7;

   --  true if Char is s,m,h,d,w or uppercase equivalent
   function IsMult (Char : in Character) return Boolean;

   function IsMult (Char : in Character) return Boolean is
      C      : Character;
      Retval : Boolean;
   begin
      C := Ada.Characters.Handling.To_Upper (Char);
      if C = 'S' or C = 'M' or C = 'H' or C = 'D' or C = 'W' then
         Retval := True;
      else
         Retval := False;
      end if;
      return Retval;
   end IsMult;

   --  converts the time specifier at S (BegIdx .. EndIdx) into a time value in
   --  seconds, returned in RetVal.
   --  a time specifier is either a Num or one or more Blobs.  A Num is a
   --  string
   --  of decimal digits.
   --  A Blob is a Num followed by a Mult.  A Mult is one of {s,m,h,w,d} and
   --  their upper-case equivalents.
   --  Sets Success to false if the time specifier is invalid
   procedure ConvertTimeSpec
     (S       : in RR_Type.LineFromFileType;
      BegIdx  : in RR_Type.LineLengthIndex;
      EndIdx  : in RR_Type.LineLengthIndex;
      RetVal  : out Unsigned_Types.Unsigned32;
      Success : in out Boolean)
   is separate;

end Parser_Utilities;
