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
--  Scan_String touches neither, so the extraction is not generic and Token
--  keeps every component but those two.
--
--  Scan_String and Scan_Hex_Quad are both declared in the package *body*
--  upstream; Scan_String is promoted to this spec with its contract unchanged,
--  and Scan_Hex_Quad is imported here with its contract unchanged, since the
--  loop calls it.

with JSON.Streams;

package JSON.Tokenizers
  with SPARK_Mode => On
is

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
      String_Offset : Positive := 1;
      String_Length : Natural := 0;
      Boolean_Value : Boolean := False;
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

   procedure Scan_Hex_Quad
     (Stream : in out Streams.Stream;
      Count  : in out Natural;
      Unit   : out Natural;
      Status : out Token_Status)
   with
     Import,
     Global => null,
     Pre    =>
       not Streams.Has_Buffered_Character (Stream)
       and then Count <= Streams.Position (Stream) - 1,
     Always_Terminates,
     Post   =>
       Streams.Length (Stream) = Streams.Length (Stream)'Old
       and then not Streams.Has_Buffered_Character (Stream)
       and then Streams.Position (Stream) >= Streams.Position (Stream)'Old
       and then Count <= Streams.Position (Stream) - 1
       and then (if Status = Success then Unit <= 16#FFFF#);
   --  Scan the 4 hexadecimal digits of a \uXXXX escape sequence and
   --  return their value as a UTF-16 code unit

   procedure Scan_String
     (Stream     : in out Streams.Stream;
      Next_Token : out Token;
      Status     : out Token_Status)
   with
     Pre  => not Streams.Has_Buffered_Character (Stream),
     Always_Terminates,
     Post =>
       Streams.Length (Stream) = Streams.Length (Stream)'Old
       and then not Streams.Has_Buffered_Character (Stream)
       and then Streams.Position (Stream) >= Streams.Position (Stream)'Old
       and then
         (if Status = Success
          then
            Next_Token.Kind = String_Token
            and then
              Streams.Valid_Span
                (Stream, Next_Token.String_Offset, Next_Token.String_Length));
   --  Scan a JSON string literal, the opening '"' having been read, and
   --  return the span of the stream it occupies in JSON escaped form

end JSON.Tokenizers;
