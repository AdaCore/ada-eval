--  SPDX-License-Identifier: Apache-2.0
--
--  Copyright (c) 2016 onox <denkpadje@gmail.com>
--
--  Licensed under the Apache License, Version 2.0 (the "License");
--  you may not use this file except in compliance with the License.
--  You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
--  Unless required by applicable law or agreed to in writing, software
--  distributed under the License is distributed on an "AS IS" BASIS,
--  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
--  See the License for the specific language governing permissions and
--  limitations under the License.

--  Extracted from json-spark, json/src/json-tokenizers.ads and .adb.
--
--  Upstream JSON.Tokenizers is generic in an instance of JSON.Types, whose two
--  numeric formals type the Integer_Value and Float_Value components of Token.
--  Scan_Literal touches neither, so the extraction is not generic and Token
--  keeps every component but those two.
--
--  Scan_Literal is declared in the package *body* upstream; it is promoted to
--  this spec with its contract unchanged.

with JSON.Streams;

package JSON.Tokenizers with SPARK_Mode => On is

   pragma Preelaborate;

   type Token_Kind is
     (Begin_Array_Token,
      Begin_Object_Token,
      End_Array_Token,
      End_Object_Token,
      Name_Separator_Token,
      Value_Separator_Token,
      String_Token,
      Integer_Token,
      Float_Token,
      Boolean_Token,
      Null_Token,
      EOF_Token,
      Invalid_Token);
   --  The kind of a JSON token produced by the tokenizer

   type Token is record
      Kind          : Token_Kind := Invalid_Token;
      String_Offset : Positive   := 1;
      String_Length : Natural    := 0;
      Boolean_Value : Boolean    := False;
   end record;
   --  Only the components corresponding to Kind are meaningful:
   --  String_Offset and String_Length for String_Token (a span of the
   --  stream in JSON escaped form), and Boolean_Value for Boolean_Token

   type Token_Status is
     (Success,
      Error_Unexpected_EOF,
      Error_Expected_EOF,
      Error_Unexpected_Character,
      Error_Control_Character,
      Error_Escaped_Character,
      Error_Escaped_Unicode,
      Error_Plus_Sign,
      Error_Leading_Zeroes,
      Error_Minus_Digit,
      Error_Dot_Digit,
      Error_Exponent_Digit,
      Error_Exponent_Sign_Digit,
      Error_Number_Too_Long,
      Error_Number_Out_Of_Range,
      Error_Invalid_Literal);
   --  The outcome of scanning a token: Success or a specific error code

   procedure Scan_Literal
     (Stream     : in out Streams.Stream;
      First      : Character;
      Next_Token : out Token;
      Status     : out Token_Status)
   with Pre  => not Streams.Has_Buffered_Character (Stream),
        Always_Terminates,
        Post =>
     Streams.Length (Stream) = Streams.Length (Stream)'Old
       and then Streams.Position (Stream) >= Streams.Position (Stream)'Old
       and then (if Streams.Has_Buffered_Character (Stream) then
                   Streams.Position (Stream) > Streams.Position (Stream)'Old)
       and then (if Status = Success then
                   Next_Token.Kind in Boolean_Token | Null_Token);
   --  Scan an identifier, its first character having been read, and return
   --  the literal true, false or null it spells

end JSON.Tokenizers;
