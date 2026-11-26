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
--  Base_Integer, and the enclosing "if", are upstream's; they are declared
--  inside Scan_Number's body upstream and had to move to the spec here so the
--  profile and the contract can name them.

with JSON.Types;

generic
   with package Types is new JSON.Types (<>);
package JSON.Tokenizers with SPARK_Mode => On is

   pragma Preelaborate;

   use type Types.Integer_Type;
   --  Upstream's body carries this clause; the contract below needs it in the
   --  visible part

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

   procedure Scan_Number_Integer_Exponent
     (Int_Acc   : in out Base_Integer;
      Exp_Value : Natural;
      Status    : out Token_Status)
   with Pre => Int_Acc <= 0, Always_Terminates, Post => Int_Acc <= 0;
   --  Multiply the accumulated integer mantissa by ten Exp_Value times,
   --  reporting Error_Number_Out_Of_Range if that would leave the range of
   --  Base_Integer.

end JSON.Tokenizers;
