"""
Hydrex Voting Optimizer - Core Algorithm (Partner-Limited, No hist_fees)
Shows POOL NAMES first (BRETT-WETH) instead of addresses
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HydrexVotingOptimizer:
    """
    Production-ready voting optimizer for Hydrex Protocol.
    Partner-limited to avoid selling pressure on partner tokens.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_pools': 8,
            'max_pool_allocation_pct': 0.15,
            'min_pool_allocation': 1000,
            'saturation_threshold': 0.50,
            'risk_weight': 0.3,
            'total_system_votes': 164e6
        }
        
        # BACKWARD COMPATIBILITY - Works with ANY config order/names
        if 'max_partner_pools' not in self.config:
            self.config['max_partner_pools'] = 2
        if 'diversification_max_same_type' in self.config:
            logger.info(f"⚠️  Removed old 'diversification_max_same_type': {self.config.pop('diversification_max_same_type')}")
        
        logger.info(f"Optimizer initialized with config: {self.config}")
    
    def classify_pool_type(self, pool: pd.Series) -> str:
        """Classify pools as 'partner' or 'non_partner'."""
        pool_id = str(pool.get('pool_id', '')).lower()
        pool_name = str(pool.get('pool_name', '')).lower()
        
        partner_keywords = [
            'aave', 'compound', 'maker', 'crv', 'convex', 'yearn', 'bal', 'sushi',
            '1inch', 'curve', 'uniswap', 'usdc', 'usdt', 'dai', 'weth', 'wbeth'
        ]
        
        for keyword in partner_keywords:
            if keyword in pool_id or keyword in pool_name:
                return 'partner'
        return 'non_partner'
    
    def compute_risk_factor(self, pool: pd.Series) -> float:
        """Calculate risk-adjusted multiplier WITHOUT hist_fees."""
        try:
            vote_share = pool['current_votes'] / self.config['total_system_votes']
            saturation_penalty = max(0, (vote_share - self.config['saturation_threshold']) * 2)
            saturation_penalty = min(saturation_penalty, 0.5)
            
            tvl_confidence = min(1.0, (pool['tvl_usd'] / 1e6) ** 0.5)
            risk_factor = (1 - saturation_penalty) * tvl_confidence
            return max(0.1, min(1.0, risk_factor))
            
        except Exception as e:
            logger.warning(f"Error computing risk for {pool.get('pool_id')}: {e}")
            return 0.5
    
    def compute_marginal_score(self, pool: pd.Series, proposed_vote: float, 
                               current_allocation: float = 0) -> float:
        """Compute marginal fee capture score."""
        try:
            total_new_votes = max(pool['current_votes'] + current_allocation + proposed_vote, 1)
            vote_share = proposed_vote / total_new_votes
            expected_fees = pool['projected_rewards'] * vote_share
            marginal_return = expected_fees / (proposed_vote + 1e-6)
            
            risk_factor = pool.get('risk_factor', 1.0)
            rewards_per_vote = pool['projected_rewards'] / (pool['current_votes'] + 1)
            undervalue_boost = 1 + np.log1p(rewards_per_vote / 0.01)
            
            return marginal_return * risk_factor * undervalue_boost
            
        except Exception as e:
            logger.warning(f"Error computing score for {pool.get('pool_id')}: {e}")
            return 0
    
    def check_constraints(self, pool: pd.Series, allocation: Dict[str, float],
                         proposed_vote: float, total_user_votes: float, 
                         df_pools: pd.DataFrame = None) -> bool:
        """Verify allocation with PARTNER POOL LIMIT."""
        pool_id = pool['pool_id']
        pool_type = pool.get('pool_type', 'non_partner')
        current_alloc = allocation.get(pool_id, 0)
        
        if current_alloc + proposed_vote > total_user_votes * self.config['max_pool_allocation_pct']:
            return False
        if proposed_vote < self.config['min_pool_allocation']:
            return False
        
        if pool_type == 'partner' and df_pools is not None:
            current_partner_count = sum(
                1 for pid, votes in allocation.items()
                if votes > 0 and not df_pools[df_pools['pool_id'] == pid].empty 
                and df_pools[df_pools['pool_id'] == pid].iloc[0]['pool_type'] == 'partner'
            )
            if current_partner_count >= self.config['max_partner_pools'] and pool_id not in allocation:
                logger.info(f"Blocked {pool_id}: max_partner_pools ({self.config['max_partner_pools']}) reached")
                return False
        
        return True
    
    def optimize_allocation(self, df_pools: pd.DataFrame, 
                           user_voting_power: float) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Main optimization function with partner limiting."""
        logger.info(f"Starting optimization with {user_voting_power:,.0f} voting power")
        
        df_pools = self._prepare_data(df_pools)
        self.config['total_system_votes'] = df_pools['current_votes'].sum()
        
        partner_count = len(df_pools[df_pools['pool_type']=='partner'])
        logger.info(f"Found {partner_count} partner pools, max allowed: {self.config['max_partner_pools']}")
        
        df_pools['risk_factor'] = df_pools.apply(self.compute_risk_factor, axis=1)
        df_pools['base_score'] = df_pools.apply(
            lambda p: self.compute_marginal_score(p, user_voting_power * 0.1), axis=1
        )
        
        candidates = df_pools.nlargest(min(15, len(df_pools)), 'base_score').copy()
        logger.info(f"Selected {len(candidates)} candidate pools")
        
        allocation = {}
        remaining_votes = user_voting_power
        
        for iteration in range(20):
            if len(allocation) >= self.config['max_pools'] or remaining_votes <= self.config['min_pool_allocation']:
                break
            
            best_pool_id = None
            best_score = -np.inf
            best_delta = 0
            
            for _, pool in candidates.iterrows():
                pool_id = pool['pool_id']
                pools_left = self.config['max_pools'] - len(allocation)
                avg_allocation = remaining_votes / max(1, pools_left)
                proposed_delta = min(
                    avg_allocation,
                    user_voting_power * self.config['max_pool_allocation_pct'] - allocation.get(pool_id, 0),
                    remaining_votes
                )
                
                if proposed_delta < self.config['min_pool_allocation']:
                    continue
                
                if not self.check_constraints(pool, allocation, proposed_delta, user_voting_power, df_pools):
                    continue
                
                score = self.compute_marginal_score(pool, proposed_delta, allocation.get(pool_id, 0))
                if score > best_score:
                    best_score = score
                    best_pool_id = pool_id
                    best_delta = proposed_delta
            
            if best_pool_id is None:
                logger.info("No more valid allocations found")
                break
            
            allocation[best_pool_id] = allocation.get(best_pool_id, 0) + best_delta
            remaining_votes -= best_delta
            pool_type = df_pools[df_pools['pool_id']==best_pool_id]['pool_type'].iloc[0]
            logger.info(f"Iteration {iteration+1}: [{pool_type.upper()}] {best_delta:,.0f} → {best_pool_id}")
        
        allocation = self._rebalance(allocation, user_voting_power)
        results_df = self._create_results_dataframe(df_pools, allocation, user_voting_power)
        
        partner_selected = sum(1 for pool_id, votes in allocation.items() 
                              if votes > 0 and not df_pools[df_pools['pool_id'] == pool_id].empty 
                              and df_pools[df_pools['pool_id'] == pool_id].iloc[0]['pool_type'] == 'partner')
        logger.info(f"Optimization complete: {len([v for v in allocation.values() if v>0])} pools ({partner_selected}/{self.config['max_partner_pools']} partners)")
        
        return allocation, results_df
    
    def _prepare_data(self, df_pools: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data - NO hist_fees processing."""
        df = df_pools.copy()
        
        required_cols = ['pool_id', 'tvl_usd', 'projected_rewards', 'current_votes']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df['pool_type'] = df.apply(self.classify_pool_type, axis=1)
        return df[df['tvl_usd'] > 0][df['projected_rewards'] > 0][df['current_votes'] >= 0]
    
    def _rebalance(self, allocation: Dict[str, float], target_total: float) -> Dict[str, float]:
        """Proportionally adjust allocation to match target total."""
        current_total = sum(allocation.values())
        if current_total == 0 or abs(current_total - target_total) < 1:
            return allocation
        scale_factor = target_total / current_total
        rebalanced = {pid: votes * scale_factor for pid, votes in allocation.items()}
        logger.info(f"Rebalanced: {current_total:,.0f} → {sum(rebalanced.values()):,.0f}")
        return rebalanced
    
    def _create_results_dataframe(self, df_pools: pd.DataFrame, 
                                  allocation: Dict[str, float],
                                  total_votes: float) -> pd.DataFrame:
        """Create detailed results - SHOWS POOL NAMES FIRST."""
        results = []
        for pool_id, votes in allocation.items():
            if pool_id not in df_pools['pool_id'].values:
                continue
            pool = df_pools[df_pools['pool_id'] == pool_id].iloc[0]
            
            total_pool_votes = pool['current_votes'] + votes
            vote_share = votes / total_pool_votes
            projected_fees = pool['projected_rewards'] * vote_share
            marginal_apr = (projected_fees / votes) * 52 * 100 if votes > 0 else 0
            
            # SHOW POOL NAME FIRST (BRETT-WETH), fallback to pool_id prefix
            display_name = (pool.get('pool_name') or 
                          (pool_id.split('-')[0] if '-' in pool_id else pool_id[:12]))
            
            results.append({
                'pool_id': pool_id,
                'pool_type': pool['pool_type'],
                'pair': display_name,  # POOL NAME DISPLAYED FIRST
                'votes_allocated': votes,
                'allocation_pct': (votes / total_votes) * 100,
                'projected_fees': projected_fees,
                'marginal_apr': marginal_apr,
                'risk_score': pool.get('risk_factor', 0),
                'timestamp': datetime.now().isoformat()
            })
        
        return pd.DataFrame(results).sort_values('votes_allocated', ascending=False)
    
    def simulate_returns(self, df_pools: pd.DataFrame, allocation: Dict[str, float],
                        n_simulations: int = 1000) -> Dict:
        """Monte Carlo simulation of expected returns."""
        logger.info(f"Running {n_simulations} simulations")
        results = []
        for _ in range(n_simulations):
            total_fees = sum(
                df_pools[df_pools['pool_id'] == pid].iloc[0]['projected_rewards'] * 
                np.random.uniform(0.8, 1.2) * (votes / (df_pools[df_pools['pool_id'] == pid].iloc[0]['current_votes'] + votes))
                for pid, votes in allocation.items() if votes > 0
            )
            results.append(total_fees)
        
        return {
            'mean': np.mean(results),
            'std': np.std(results),
            'median': np.median(results),
            'p5': np.percentile(results, 5),
            'p95': np.percentile(results, 95),
            'sharpe': np.mean(results) / (np.std(results) + 1e-6)
        }

if __name__ == "__main__":
    print("Optimizer module loaded successfully")
