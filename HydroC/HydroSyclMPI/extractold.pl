#!/usr/bin/env perl
BEGIN {
     $elaps = 0.;
}
while ( <> ) {
   @a=split(" ");
   if ( /max threads/ ) {
      $threads=$a[2];
   }
   if ( /Hydro ends/ ) {
      $elapst=$a[4];
      $elapst =~ s/\(//;
      $elapst =~ s/\)//;
      $elaps = $elapst;
   }
}
END {
   if ( $elaps > 0. ) { print $threads . " " . $elaps ."\n"; };
}
