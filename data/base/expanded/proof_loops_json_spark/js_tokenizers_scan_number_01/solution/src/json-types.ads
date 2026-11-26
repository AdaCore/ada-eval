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

--  Extracted from json-spark, json/src/json-types.ads.
--
--  Only the part of JSON.Types that Scan_Number names: the two numeric formal
--  types, whose base types the number accumulators are declared from, and the
--  bound on the textual length of a number.  The JSON value tree --
--  JSON_Value,
--  its access types, its constructors and Free -- is the rest of the package,
--  and the number scanner touches none of it.
--
--  The generic formals are copied unchanged, including the default for
--  Maximum_Number_Length, because the extraction instantiates this package
--  with
--  upstream's own actuals (see instances.ads).

pragma Warnings (Off, "type ""Float_Type"" is not referenced");
pragma Warnings (Off, "type ""Integer_Type"" is not referenced");
generic
   type Integer_Type is range <>;
   type Float_Type is digits <>;

   Maximum_Number_Length : Positive := 30;
package JSON.Types with SPARK_Mode => On is
   --  @formal Integer_Type The integer type used for JSON integer values
   --  @formal Float_Type The floating-point type used for JSON float
   --    values
   --  @formal Maximum_Number_Length Maximum length, in characters, of the
   --    textual form of a number accepted while parsing

   pragma Preelaborate;

   Maximum_String_Length_Numbers : constant Positive := Maximum_Number_Length;
   --  Maximum length, in characters, of the textual form of a number

end JSON.Types;
