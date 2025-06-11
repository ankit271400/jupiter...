import pandas as pd
            # Generate features independently
            for feature, config in self.feature_configs.items():
                if feature not in df.columns:
                    if 'mean' in config and 'std' in config:
                        df[feature] = np.random.normal(
                            config['mean'], config['std'], self.n_samples
                        ).clip(config['min'], config['max'])
                    else:
                        df[feature] = np.random.uniform(
                            config['min'], config['max'], self.n_samples
                        )
        
        # Add outliers
        df = self._add_outliers(df)
        
        # Create target variable
        df = self._create_target_variable(df)
        
        # Final data cleaning and validation
        df = self._clean_and_validate(df)
        
        return df
    
    def _clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the generated dataset."""
        
        # Ensure no negative values where they shouldn't exist
        non_negative_cols = ['monthly_income', 'monthly_emi_outflow', 'current_outstanding',
                            'credit_utilization_ratio', 'num_open_loans', 'repayment_history_score',
                            'dpd_last_3_months', 'num_hard_inquiries_last_6m',
                            'recent_credit_card_usage', 'recent_loan_disbursed_amount',
                            'total_credit_limit', 'months_since_last_default']
        
        for col in non_negative_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        # Ensure credit utilization ratio is between 0 and 1
        df['credit_utilization_ratio'] = df['credit_utilization_ratio'].clip(0, 1)
        
        # Ensure repayment history score is between 0 and 100
        df['repayment_history_score'] = df['repayment_history_score'].clip(0, 100)
        
        # Round numerical columns appropriately
        int_columns = ['age', 'monthly_income', 'monthly_emi_outflow', 'current_outstanding',
                      'num_open_loans', 'repayment_history_score', 'dpd_last_3_months',
                      'num_hard_inquiries_last_6m', 'recent_credit_card_usage',
                      'recent_loan_disbursed_amount', 'total_credit_limit',
                      'months_since_last_default']
        
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].round().astype(int)
        
        # Round credit utilization ratio to 3 decimal places
        df['credit_utilization_ratio'] = df['credit_utilization_ratio'].round(3)
        
        return df
