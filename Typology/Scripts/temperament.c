/*
 *
 * SPDX-License-Identifier: CC-BY-4.0
 * Copyright (c) 2025 Gourav Kumar Mallick
 *
 * Licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
 * You may copy, redistribute, remix, transform, and build upon this work for any purpose,
 * even commercially, provided that you give appropriate credit to the original author,
 * include a link to the license, and indicate if changes were made.
 *
 * Full license: https://creativecommons.org/licenses/by/4.0/
 *
 */










/* Importing all required libraries. */

#include <stdio.h>
#include <stdbool.h>





/* Fixing macros. */

#define N 5			// Population size (total involved biochemicals)
#define C 0			// Cortisol index
#define D 1			// Dopamine index
#define O 2			// Oxytocin index
#define S 3			// Serotonin index
#define A 4			// Androgenicity (estrogen/testosterone) index

#define CHOL 0		// Integer representative for choleric type.
#define MEL 1		// Integer representative for melancholic type.
#define PHLEG 2		// Integer representative for phlegmatic type.
#define SANG 3		// Integer representative for sanguine type.
#define TOTAL 4		// Total number of temperaments.










/* Making global storehouses. */

signed short int chol[N], mel[N], phleg[N], sang[N];
signed short int user[N] = {0};
signed short int chol_diff[N], mel_diff[N], phleg_diff[N], sang_diff[N];
float variances[4];










/* Computes variance for given conditions. */

float variance(signed short int *temp_diff)
{
	float mean = (float)((*(temp_diff+C)) + (*(temp_diff+D)) + (*(temp_diff+O)) + (*(temp_diff+S)) + (*(temp_diff+A)))/(float)N;
	float sum = 0.0;



	/* Summing all biochemical scores for variance summation. */

	sum += ((float)(*(temp_diff+C)) - mean)*((float)(*(temp_diff+C)) - mean);
	sum += ((float)(*(temp_diff+D)) - mean)*((float)(*(temp_diff+D)) - mean);
	sum += ((float)(*(temp_diff+O)) - mean)*((float)(*(temp_diff+O)) - mean);
	sum += ((float)(*(temp_diff+S)) - mean)*((float)(*(temp_diff+S)) - mean);
	sum += ((float)(*(temp_diff+A)) - mean)*((float)(*(temp_diff+A)) - mean);



	/* Returning the variance result. */

	return (sum/(float)N);
}










/* Main program. */

int main()
{
	/* License terms. */

	printf("\nSPDX-License-Identifier: CC-BY-4.0\n");
	printf("Copyright (c) 2025 Gourav Kumar Mallick\n\n\n");





	/* Command line information for users. */

	printf("Rating points meaning:\n");
	printf("1: Extreme deficiency\n");
	printf("2: Mild deficiency\n");
	printf("3: Balanced levels\n");
	printf("4: Slight excessiveness\n");
	printf("5: Highly excessive\n\n");





	/* Ideal biochemical points for each type. */

	chol[C] = 3; chol[D] = 5; chol[O] = 1; chol[S] = 2; chol[A] = 5;
	mel[C] = 5; mel[D] = 1; mel[O] = 3; mel[S] = 4; mel[A] = 1;
	phleg[C] = 1; phleg[D] = 2; phleg[O] = 5; phleg[S] = 5; phleg[A] = 1;
	sang[C] = 1; sang[D] = 5; sang[O] = 3; sang[S] = 3; sang[A] = 3;






	/* Creating user profile. */

	printf("ANDROGENICITY for males = TESTOSTERONE\n");
	printf("ANDROGENICITY for females = ESTROGEN\n\n");



	while (!(user[C]>=1 && user[C]<=5))
	{
		printf("Rate your usual CORTISOL (C) levels: ");
		scanf("%hd", &user[C]);

		if (user[C]<1 || user[C]>5)
		{
			printf("Your CORTISOL score was out of range [1,5].\n");
			printf("Please re-enter, or exit with \"CTRL + C\".\n\n");
		}
	}

	 

	while (!(user[D]>=1 && user[D]<=5))
	{
		printf("Rate your usual DOPAMINE (D) levels: ");
		scanf("%hd", &user[D]);

		if (user[D]<1 || user[D]>5)
		{
			printf("Your DOPAMINE score was out of range [1,5].\n");
			printf("Please re-enter, or exit with \"CTRL + C\".\n\n");
		}
	}



	while (!(user[O]>=1 && user[O]<=5))
	{
		printf("Rate your usual OXYTOCIN (O) levels: ");
		scanf("%hd", &user[O]);

		if (user[O]<1 || user[O]>5)
		{
			printf("Your OXYTOCIN score was out of range [1,5].\n");
			printf("Please re-enter, or exit with \"CTRL + C\".\n\n");
		}
	}

	

	while (!(user[S]>=1 && user[S]<=5))
	{
		printf("Rate your usual SEROTONIN (S) levels: ");
		scanf("%hd", &user[S]);

		if (user[S]<1 || user[S]>5)
		{
			printf("Your SEROTONIN score was out of range [1,5].\n");
			printf("Please re-enter, or exit with \"CTRL + C\".\n\n");
		}
	}

	

	while (!(user[A]>=1 && user[A]<=5))
	{
		printf("Rate your usual ANDROGENICITY (A) levels: ");
		scanf("%hd", &user[A]);

		if (user[A]<1 || user[A]>5)
		{
			printf("Your ANDROGENICITY score was out of range [1,5].\n");
			printf("Please re-enter, or exit with \"CTRL + C\".\n\n");
		}
	}





	printf("\nComputing results...\n");





	/* Computing differences with choleric type. */

	chol_diff[C] = user[C] - chol[C];
	chol_diff[D] = user[D] - chol[D];
	chol_diff[O] = user[O] - chol[O];
	chol_diff[S] = user[S] - chol[S];
	chol_diff[A] = user[A] - chol[A];



	/* Computing differences with melancholic type. */

	mel_diff[C] = user[C] - mel[C];
	mel_diff[D] = user[D] - mel[D];
	mel_diff[O] = user[O] - mel[O];
	mel_diff[S] = user[S] - mel[S];
	mel_diff[A] = user[A] - mel[A];



	/* Computing differences with phlegmatic type. */

	phleg_diff[C] = user[C] - phleg[C];
	phleg_diff[D] = user[D] - phleg[D];
	phleg_diff[O] = user[O] - phleg[O];
	phleg_diff[S] = user[S] - phleg[S];
	phleg_diff[A] = user[A] - phleg[A];



	/* Computing differences with sanguine type. */

	sang_diff[C] = user[C] - sang[C];
	sang_diff[D] = user[D] - sang[D];
	sang_diff[O] = user[O] - sang[O];
	sang_diff[S] = user[S] - sang[S];
	sang_diff[A] = user[A] - sang[A];





	/* Computing variance for each temperament. */

	variances[CHOL] = variance(chol_diff); printf("Variance for CHOLERIC type: %f\n", variances[CHOL]);
	variances[MEL] = variance(mel_diff); printf("Variance for MELANCHOLIC type: %f\n", variances[MEL]);
	variances[PHLEG] = variance(phleg_diff); printf("Variance for PHLEGMATIC type: %f\n", variances[PHLEG]);
	variances[SANG] = variance(sang_diff); printf("Variance for SANGUINE type: %f\n\n", variances[SANG]);





	/* Finding the primary & secondary temperament. */

	float first, second;
	int primary, secondary;


	if (variances[1]<=variances[0])
	{
		second = variances[0]; secondary = 0;
		first = variances[1]; primary = 1;
	}
	else if (variances[1]>variances[0])
	{
		second = variances[1]; secondary = 1;
		first = variances[0]; primary = 0;
	}



	for (int i=2; i<TOTAL; i++)
	{
		if (variances[i]<=first)
		{
			second = first; secondary = primary;
			first = variances[i]; primary = i;
		}
		else if (variances[i]<second) {second = variances[i]; secondary = i;}
	}





	/* Checking for strength of blend. */

	bool blend;

	if (second-first<0.5) {blend = true; printf("Strong blend detected (%f < 0.5).\n", second-first);}
	else {blend = false; printf("Blend is weak (%f >= 0.5).\n", second-first);}





	/* Finalizing results. */

	printf("RESULT: ");



	switch (primary)
	{
		case CHOL: printf("Choleric"); break;
		case MEL: printf("Melancholic"); break;
		case PHLEG: printf("Phlegmatic"); break;
		case SANG: printf("Sanguine"); break;
	}



	if (blend==false) {printf(" [Dominant]\n\n");}
	else if (blend==true)
	{
		switch (secondary)
		{
			case CHOL: printf("-Choleric\n\n"); break;
			case MEL: printf("-Melancholic\n\n"); break;
			case PHLEG: printf("-Phlegmatic\n\n"); break;
			case SANG: printf("-Sanguine\n\n"); break;
		}
	}





	return 0;
}