package PATTERN;

import java.util.*;

public class pattern1{
public static void main(String[] args){

Scanner sc = new Scanner (System.in);
int n = sc.nextInt();
int m = sc.nextInt();

for(int i = 0 ; i<n ; i++){
    for(int j = 0; j <m ; j++){
        System.out.print("*");
    }
    System.out.print("\n");
}
sc.close();

}
}




//print star 
// ****
// ****
// ****
// ****