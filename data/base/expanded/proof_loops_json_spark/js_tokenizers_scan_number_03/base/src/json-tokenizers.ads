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
--  Base_Integer, Base_Float, Mantissa_Limit and Exp_Limit are upstream's own
--  declarations from inside Scan_Number's body, moved to this spec so the
--  profile and the contract can name them.

with JSON.Types;
with JSON.Streams;

generic
   with package Types is new JSON.Types (<>);
package JSON.Tokenizers with SPARK_Mode => On is

   pragma Preelaborate;

   use type Types.Integer_Type;
   use type Types.Float_Type;
   --  Upstream's body carries these clauses; the declarations and the contract
   --  below need them in the visible part

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

   subtype Base_Integer is Types.Integer_Type'Base;
   subtype Base_Float is Types.Float_Type'Base;

   Mantissa_Limit : constant Base_Float := Base_Float'Last / 16.0;
   --  Base_Float'Last / 16.0 is exact (division by a power of two),
   --  which makes the overflow guards below provable

   Exp_Limit : constant := 100_000;
   --  Beyond this value the exponent saturates: the result is the
   --  same (zero, or out of range) for all larger exponents, because
   --  the mantissa has at most Maximum_Number_Length digits

   procedure Scan_Number_Exponent_Digits
     (Stream     : in out Streams.Stream;
      Char_Count : in out Natural;
      Int_Acc    : Base_Integer;
      F_Acc      : Base_Float;
      Exp_Value  : in out Natural;
      Status     : out Token_Status)
   with
     Pre  =>
       not Streams.Has_Buffered_Character (Stream)
       and then Char_Count <= Types.Maximum_String_Length_Numbers
       and then Int_Acc <= 0
       and then F_Acc in 0.0 .. Mantissa_Limit * 10.0 + 9.0
       and then Exp_Value <= Exp_Limit,
     Always_Terminates,
     Post =>
       not Streams.Has_Buffered_Character (Stream)
       and then Streams.Length (Stream) = Streams.Length (Stream)'Old
       and then Streams.Position (Stream) >= Streams.Position (Stream)'Old
       and then Char_Count <= Types.Maximum_String_Length_Numbers
       and then Exp_Value <= Exp_Limit;
   --  Scan the digits of the exponent of a JSON number, the 'e', any sign and
   --  the first digit having been consumed, and accumulate them into
   --  Exp_Value,
   --  saturating at Exp_Limit.

end JSON.Tokenizers;
