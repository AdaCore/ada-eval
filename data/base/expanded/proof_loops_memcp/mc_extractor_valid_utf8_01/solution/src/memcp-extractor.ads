--  Extracted from memcp, src/memcp-extractor.adb: Valid_Utf8 and nothing else.
--
--  The rest of upstream's spec -- Turn, Turn_Vectors, Transcript_Ptr,
--  Decode_Base64, Extract_Turns, Extract_Recap -- is deleted, together with
--  the body-local Sextet, Strip and the json machinery: none of it is
--  reachable
--  from this loop, and keeping it would change the check count without
--  changing the loop.

package Memcp.Extractor
  with SPARK_Mode => On
is

   function Valid_Utf8 (S : String) return Boolean
   with Pre => S'Last <= Natural'Last - 4;
   --  True when S, taken as one octet per Character, is well-formed UTF-8
   --  per RFC 3629: overlong forms, surrogates U+D800 .. DFFF and code
   --  points above U+10FFFF are all rejected.
   --
   --  The precondition is authored: upstream's only call passes
   --  Memcp.Text.Value (B), whose Post gives 'First = 1 and 'Length =
   --  Length (B) <= Max_Len = Natural'Last / 8, so no index can come near
   --  Natural'Last.  It is restated here as the weakest fact the loop
   --  actually needs -- the index advances by at most four per iteration --
   --  rather than as the caller's much stronger length bound.

end Memcp.Extractor;
