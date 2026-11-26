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

--  Extracted from json-spark, json/src/json-tokenizers.ads.

with JSON.Types;
with JSON.Streams;

generic
   with package Types is new JSON.Types (<>);
package JSON.Tokenizers with SPARK_Mode => On is
   --  Tokenizer that turns JSON text into a stream of tokens.
   --  @formal Types Instance of JSON.Types providing the Integer_Type and
   --    Float_Type used to hold numeric token values

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
   --
   --  @enum Begin_Array_Token The '[' that opens an array
   --  @enum Begin_Object_Token The '{' that opens an object
   --  @enum End_Array_Token The ']' that closes an array
   --  @enum End_Object_Token The '}' that closes an object
   --  @enum Name_Separator_Token The ':' between a member name and its value
   --  @enum Value_Separator_Token The ',' between array elements or object
   --  members
   --  @enum String_Token A string literal
   --  @enum Integer_Token A number with no fractional or exponent part
   --  @enum Float_Token A number with a fractional and/or exponent part
   --  @enum Boolean_Token The literal true or false
   --  @enum Null_Token The literal null
   --  @enum EOF_Token The end of the stream
   --  @enum Invalid_Token No valid token; used as the default kind

   type Token is record
      Kind          : Token_Kind := Invalid_Token;
      String_Offset : Positive := 1;
      String_Length : Natural := 0;
      Integer_Value : Types.Integer_Type := Types.Integer_Type'First;
      Float_Value   : Types.Float_Type := Types.Float_Type'First;
      Boolean_Value : Boolean := False;
   end record;
   --  Only the components corresponding to Kind are meaningful:
   --  String_Offset and String_Length for String_Token (a span of the
   --  stream in JSON escaped form), Integer_Value for Integer_Token,
   --  Float_Value for Float_Token, and Boolean_Value for Boolean_Token
   --
   --  @field Kind The kind of token this record holds
   --  @field String_Offset Start position in the stream of a String_Token's
   --    span, in JSON escaped form
   --  @field String_Length Length of a String_Token's span, in JSON escaped
   --  form
   --  @field Integer_Value Value of an Integer_Token
   --  @field Float_Value Value of a Float_Token
   --  @field Boolean_Value Value of a Boolean_Token

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
   --
   --  @enum Success A token was scanned successfully
   --  @enum Error_Unexpected_EOF The stream ended in the middle of a token
   --  @enum Error_Expected_EOF More input was found where the end of the
   --    stream was expected
   --  @enum Error_Unexpected_Character A character was found that cannot
   --    start or occur in a token
   --  @enum Error_Control_Character An unescaped control character occurred
   --    inside a string
   --  @enum Error_Escaped_Character An invalid escape sequence occurred inside
   --    a string
   --  @enum Error_Escaped_Unicode An invalid \u Unicode escape occurred inside
   --    a string
   --  @enum Error_Plus_Sign A '+' sign appeared where it is not permitted in
   --    a number
   --  @enum Error_Leading_Zeroes A number had disallowed leading zeroes
   --  @enum Error_Minus_Digit A '-' sign was not followed by a digit
   --  @enum Error_Dot_Digit A decimal point was not followed by a digit
   --  @enum Error_Exponent_Digit An exponent had no digits
   --  @enum Error_Exponent_Sign_Digit An exponent sign was not followed by a
   --    digit
   --  @enum Error_Number_Too_Long A number had too many characters to be
   --    represented
   --  @enum Error_Number_Out_Of_Range A number could not be represented by the
   --    target numeric type
   --  @enum Error_Invalid_Literal An identifier was not one of true, false or
   --    null

   procedure Scan_Token
     (Stream     : in out Streams.Stream;
      Next_Token : out Token;
      Status     : out Token_Status;
      Expect_EOF : Boolean := False)
   with
     Import,
     Global => null,
     Always_Terminates,
     Post   =>
       Streams.Length (Stream) = Streams.Length (Stream)'Old
       and then Streams.Position (Stream) >= Streams.Position (Stream)'Old
       and then
         (if Streams.Has_Buffered_Character (Stream)
            and then Streams.Position (Stream) = Streams.Position (Stream)'Old
          then Streams.Has_Buffered_Character (Stream)'Old)
       and then
         (if Status = Success
          then
            Next_Token.Kind /= Invalid_Token
            and then Expect_EOF = (Next_Token.Kind = EOF_Token)
            and then
              (if Next_Token.Kind = String_Token
               then
                 Streams.Valid_Span
                   (Stream,
                    Next_Token.String_Offset,
                    Next_Token.String_Length))
            and then
              (if Next_Token.Kind /= EOF_Token
               then
                 Streams.Position (Stream) > Streams.Position (Stream)'Old
                 or else
                   (Streams.Position (Stream) = Streams.Position (Stream)'Old
                    and then Streams.Has_Buffered_Character (Stream)'Old
                    and then not Streams.Has_Buffered_Character (Stream))));
   --  Read the next token from the stream. Status reports failures
   --  instead of raising an exception, which allows a caller to release
   --  memory it owns before propagating an error.
   --
   --  The postcondition guarantees progress: reading a token other than
   --  EOF_Token consumes at least one character of the stream or the
   --  buffered character. This allows callers to prove that their
   --  parsing loops terminate.
   --
   --  @param Stream The stream to read the token from
   --  @param Next_Token The token that was scanned, valid only when Status
   --    is Success
   --  @param Status Success, or the error code describing why scanning failed
   --  @param Expect_EOF Whether the end of the stream is expected next; when
   --    True a successful scan yields an EOF_Token

end JSON.Tokenizers;
