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
--  Only ConvertTimeSpec is kept; the other 20-odd subprograms of the upstream
--  spec, and the declarations only they referenced, are dropped.  The
--  declaration below, its Depends aspect included, is upstream's verbatim.

with RR_Type, Unsigned_Types;

package Parser_Utilities
  with SPARK_Mode
is

   --  converts the time specifier at S (BegIdx .. EndIdx) into a time value in
   --  seconds, returned in RetVal.
   procedure ConvertTimeSpec
     (S       : in RR_Type.LineFromFileType;
      BegIdx  : in RR_Type.LineLengthIndex;
      EndIdx  : in RR_Type.LineLengthIndex;
      RetVal  : out Unsigned_Types.Unsigned32;
      Success : in out Boolean)
   with Depends => ((RetVal, Success) => (BegIdx, EndIdx, S, Success));

end Parser_Utilities;
