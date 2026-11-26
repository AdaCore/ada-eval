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
--  Trimmed to the three declarations ConvertTimeSpec needs.  SPARK_Mode is
--  stated here because upstream receives it from the testsuite's generated
--  configuration pragmas file.

package RR_Type
  with SPARK_Mode
is
   MaxLineLength : constant Natural := 255 + 1;
   --  256 is what we tell the user
   subtype LineLengthIndex is Integer range 1 .. MaxLineLength;
   subtype LineFromFileType is String (LineLengthIndex);
end RR_Type;
