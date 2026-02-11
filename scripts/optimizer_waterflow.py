"""
Hydrex Voting Optimizer - WATER-FILLING VERSION (FIXED)
✅ Empty allocation safety + constraint debugging + whale-scale support
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
    Water-filling optimizer for Hydrex Protocol (FIXED VERSION).
    Allocates in small increments until marginal returns equalize.
    ✅ Handles empty allocations, whale voting power, constraint debugging
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_pools': 12,                  # ↑ Increased for whales
            'max_pool_allocation_pct': 0.20,  # ↑ 20% max per pool
            'min_pool_allocation': 5000,      # ↓ Reduced minimum
            'saturation_threshold': 0.50,
            'risk_weight': 0.3,
            'total_system_votes': 164e6,
            'max_partner_pools': 2,           # ↑ Increased for test data
            'step_size': 10000                # ↑ Larger steps for speed
        }
        
        # Backward compatibility
        if 'max_partner_pools' not in self.config:
            self.config['max_partner_pools'] = 6
        if 'step_size' not in self.config:
            self.config['step_size'] = 10000
        if 'diversification_max_same_type' in self.config:
            self.config.pop('diversification_max_same_type')
        
        logger.info(f"✅ Water-Filling Optimizer initialized (FIXED)")
        logger.info(f"   Step size: {self.config['step_size']:,} votes")
        logger.info(f"   Max partner pools: {self.config['max_partner_pools']}")
        logger.info(f"   Max pools: {self.config['max_pools']}")
    
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
    
    def compute_marginal_return(self, pool: pd.Series, proposed_vote: float, 
                                current_allocation: float = 0) -> float:
        """
        Calculate MARGINAL RETURN (fees per vote).
        This is what water-filling equalizes across pools.
        """
        try:
            total_new_votes = max(pool['current_votes'] + current_allocation + proposed_vote, 1)
            vote_share = proposed_vote / total_new_votes
            expected_fees = pool['projected_rewards'] * vote_share
            marginal_return = expected_fees / (proposed_vote + 1e-6)
            
            # Apply risk adjustment
            risk_factor = pool.get('risk_factor', 1.0)
            
            return marginal_return * risk_factor
            
        except Exception as e:
            logger.warning(f"Error computing marginal return for {pool.get('pool_id')}: {e}")
            return 0
    
    def check_constraints(self, pool: pd.Series, allocation: Dict[str, float],
                         proposed_vote: float, total_user_votes: float, 
                         df_pools: pd.DataFrame = None) -> Tuple[bool, str]:
        """Enhanced constraint checker with DEBUG info."""
        pool_id = pool['pool_id']
        pool_type = pool.get('pool_type', 'non_partner')
        current_alloc = allocation.get(pool_id, 0)
        
        constraint_failures = []
        
        # 1. Max per pool check
        max_allowed = total_user_votes * self.config['max_pool_allocation_pct']
        if current_alloc + proposed_vote > max_allowed:
            constraint_failures.append(f"MAX_POOL({max_allowed:,.0f})")
        
        # 2. Min allocation check (only for NEW pools)
        if current_alloc == 0 and proposed_vote < self.config['min_pool_allocation']:
            constraint_failures.append(f"MIN_ALLOC({self.config['min_pool_allocation']:,})")
        
        # 3. Max pools check
        if pool_id not in allocation and len(allocation) >= self.config['max_pools']:
            constraint_failures.append(f"MAX_POOLS({self.config['max_pools']})")
        
        # 4. Partner pool limit check
        if pool_type == 'partner' and df_pools is not None:
            current_partner_count = sum(
                1 for pid, votes in allocation.items()
                if votes > 0 and pid in df_pools['pool_id'].values
                and df_pools[df_pools['pool_id'] == pid].iloc[0]['pool_type'] == 'partner'
            )
            if current_partner_count >= self.config['max_partner_pools'] and pool_id not in allocation:
                constraint_failures.append(f"MAX_PARTNERS({self.config['max_partner_pools']})")
        
        can_allocate = len(constraint_failures) == 0
        return can_allocate, "; ".join(constraint_failures) if not can_allocate else "OK"
    
    def optimize_allocation_waterfilling(self, df_pools: pd.DataFrame, 
                                         user_voting_power: float) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        WATER-FILLING OPTIMIZATION (ENHANCED WITH DEBUGGING)
        """
        logger.info(f"🌊 Starting WATER-FILLING with {user_voting_power:,.0f} votes")
        logger.info(f"   Max per pool: {user_voting_power * self.config['max_pool_allocation_pct']:.0f}")
        
        df_pools = self._prepare_data(df_pools)
        self.config['total_system_votes'] = df_pools['current_votes'].sum()
        
        partner_count = len(df_pools[df_pools['pool_type']=='partner'])
        non_partner_count = len(df_pools[df_pools['pool_type']=='non_partner'])
        logger.info(f"   Pools: {partner_count} partners, {non_partner_count} non-partners")
        
        # Compute risk factors
        df_pools['risk_factor'] = df_pools.apply(self.compute_risk_factor, axis=1)
        
        # Pre-filter candidates
        df_pools['initial_score'] = df_pools.apply(
            lambda p: self.compute_marginal_return(p, user_voting_power * 0.05), axis=1
        )
        candidates = df_pools.nlargest(min(25, len(df_pools)), 'initial_score').copy()
        logger.info(f"   Pre-filtered to {len(candidates)} candidates")
        
        # 🔥 DEBUG: Test constraints on top candidates
        logger.info("   🔍 Testing constraints on top 3 candidates:")
        for idx, pool in candidates.head(3).iterrows():
            test_delta = min(50000, user_voting_power)
            can_alloc, reason = self.check_constraints(pool, {}, test_delta, user_voting_power, df_pools)
            logger.info(f"     {pool.get('pool_name', pool['pool_id'])[:25]:25s} | {reason}")
        
        # Water-filling allocation
        allocation = {}
        remaining_votes = user_voting_power
        step_size = self.config['step_size']
        iteration = 0
        max_iterations = int(user_voting_power / step_size * 2) + 100
        
        while remaining_votes >= step_size and iteration < max_iterations:
            iteration += 1
            best_pool_id = None
            best_marginal_return = -np.inf
            best_pool_info = None
            
            for idx, pool in candidates.iterrows():
                pool_id = pool['pool_id']
                current_alloc = allocation.get(pool_id, 0)
                delta_v = min(step_size, remaining_votes)
                
                can_alloc, reason = self.check_constraints(pool, allocation, delta_v, user_voting_power, df_pools)
                if not can_alloc:
                    continue
                
                marginal_return = self.compute_marginal_return(pool, delta_v, current_alloc)
                
                if marginal_return > best_marginal_return:
                    best_marginal_return = marginal_return
                    best_pool_id = pool_id
                    best_pool_info = pool
            
            if best_pool_id is None:
                logger.info(f"   No valid pools at iteration {iteration}")
                break
            
            delta = min(step_size, remaining_votes)
            allocation[best_pool_id] = allocation.get(best_pool_id, 0) + delta
            remaining_votes -= delta
            
            if iteration % 50 == 0 or remaining_votes < user_voting_power * 0.1:
                pool_type = best_pool_info['pool_type'].upper()
                logger.info(f"   Iter {iteration}: [{pool_type}] {best_pool_id[:12]}... "
                          f"{allocation[best_pool_id]:,.0f} | MR: ${best_marginal_return:.6f}")
        
        logger.info(f"   Complete: {iteration} iterations, {len(allocation)} pools")
        
        # Final rebalance
        allocation = self._rebalance(allocation, user_voting_power)
        
        # ✅ FIXED: Create results with safety check
        results_df = self._create_results_dataframe(df_pools, allocation, user_voting_power)
        
        partner_selected = sum(
            1 for pool_id in allocation.keys()
            if pool_id in df_pools['pool_id'].values
            and df_pools[df_pools['pool_id'] == pool_id].iloc[0]['pool_type'] == 'partner'
        )
        
        logger.info(f"✅ SUCCESS: {len(allocation)} pools ({partner_selected}/{self.config['max_partner_pools']} partners)")
        logger.info(f"   Total allocated: {sum(allocation.values()):,.0f}/{user_voting_power:,.0f}")
        
        return allocation, results_df
    
    # Backward compatibility
    def optimize_allocation(self, df_pools: pd.DataFrame, 
                           user_voting_power: float) -> Tuple[Dict[str, float], pd.DataFrame]:
        return self.optimize_allocation_waterfilling(df_pools, user_voting_power)
    
    def _prepare_data(self, df_pools: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data validation."""
        df = df_pools.copy()
        
        required_cols = ['pool_id', 'tvl_usd', 'projected_rewards', 'current_votes']
        optional_cols = ['pool_name', 'pool_type']
        
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise ValueError(f"❌ Missing required columns: {missing_required}")
        
        # Handle missing pool_type
        if 'pool_type' not in df.columns:
            logger.warning("⚠️  No pool_type column - creating default 'non_partner'")
            df['pool_type'] = 'non_partner'
        
        # Normalize pool_type
        valid_types = ['partner', 'non_partner']
        df['pool_type'] = df['pool_type'].astype(str).str.lower()
        df.loc[~df['pool_type'].isin(valid_types), 'pool_type'] = 'non_partner'
        
        # Filter invalid data
        df = df[df['tvl_usd'] > 0].copy()
        df = df[df['projected_rewards'] > 0].copy()
        df = df[df['current_votes'] >= 0].copy()
        
        logger.info(f"✅ Data validated: {len(df)} pools")
        return df
    
    def _rebalance(self, allocation: Dict[str, float], target_total: float) -> Dict[str, float]:
        """Proportionally adjust to exact target."""
        current_total = sum(allocation.values())
        if current_total == 0 or abs(current_total - target_total) < 100:
            return allocation
        
        scale_factor = target_total / current_total
        return {pid: max(0, votes * scale_factor) for pid, votes in allocation.items()}
    
    def _create_results_dataframe(self, df_pools: pd.DataFrame, 
                                  allocation: Dict[str, float],
                                  total_votes: float) -> pd.DataFrame:
        """✅ FIXED: Empty allocation safety + proper column handling."""
        
        # 🔥 SAFETY CHECK - Handle empty allocation
        if not allocation:
            logger.warning("⚠️  Empty allocation detected - returning empty results")
            return pd.DataFrame({
                'pool_id': [], 'pool_type': [], 'pair': [], 'votes_allocated': [],
                'allocation_pct': [], 'projected_fees': [], 'marginal_apr': [],
                'risk_score': [], 'timestamp': []
            })
        
        results = []
        for pool_id, votes in allocation.items():
            if votes == 0:
                continue
                
            if pool_id not in df_pools['pool_id'].values:
                logger.warning(f"⚠️  Pool {pool_id} not in dataframe - skipping")
                continue
            
            pool = df_pools[df_pools['pool_id'] == pool_id].iloc[0]
            
            total_pool_votes = pool['current_votes'] + votes
            vote_share = votes / total_pool_votes
            projected_fees = pool['projected_rewards'] * vote_share
            marginal_apr = (projected_fees / votes) * 52 * 100 if votes > 0 else 0
            
            display_name = (pool.get('pool_name') or 
                          str(pool_id)[:12] + '...')
            
            results.append({
                'pool_id': pool_id,
                'pool_type': pool['pool_type'],
                'pair': display_name,
                'votes_allocated': votes,
                'allocation_pct': (votes / total_votes) * 100,
                'projected_fees': projected_fees,
                'marginal_apr': marginal_apr,
                'risk_score': pool.get('risk_factor', 0),
                'timestamp': datetime.now().isoformat()
            })
        
        if not results:
            logger.warning("⚠️  No valid results generated")
            return pd.DataFrame({
                'pool_id': [], 'pool_type': [], 'pair': [], 'votes_allocated': [],
                'allocation_pct': [], 'projected_fees': [], 'marginal_apr': [],
                'risk_score': [], 'timestamp': []
            })
        
        # ✅ SAFE SORT - results guaranteed to have votes_allocated column
        df_results = pd.DataFrame(results)
        return df_results.sort_values('votes_allocated', ascending=False)
    
    def simulate_returns(self, df_pools: pd.DataFrame, allocation: Dict[str, float],
                        n_simulations: int = 1000) -> Dict:
        """Monte Carlo simulation."""
        if not allocation:
            return {'mean': 0, 'std': 0, 'p5': 0, 'p95': 0, 'sharpe': 0}
        
        logger.info(f"🎲 Running {n_simulations} simulations...")
        results = []
        
        for sim in range(n_simulations):
            total_fees = 0
            for pool_id, votes in allocation.items():
                if pool_id not in df_pools['pool_id'].values or votes == 0:
                    continue
                
                pool = df_pools[df_pools['pool_id'] == pool_id].iloc[0]
                reward_multiplier = np.random.uniform(0.8, 1.2)
                simulated_rewards = pool['projected_rewards'] * reward_multiplier
                
                total_votes = pool['current_votes'] + votes
                vote_share = votes / total_votes
                fees = simulated_rewards * vote_share
                total_fees += fees
            
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
    print("🚀 Hydrex Water-Filling Optimizer (FIXED VERSION)")
    print("✅ Empty allocation safety")
    print("✅ Constraint debugging") 
    print("✅ Whale-scale support (20% max/pool, 12 pools)")
    print("✅ Partner pool handling")
    print("\nReady for production!")
