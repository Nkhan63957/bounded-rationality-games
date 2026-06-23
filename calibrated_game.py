"""
calibrated_game.py — asymmetric US-China tariff game, empirically anchored.

Calibration sources:
  - Ossa (2014, AER 104(12):4104-46): optimal tariffs ~62%, Nash/trade-war
    tariffs ~63%, trade-war welfare loss ~2.9%. Nash via iterated best response.
  - Broda & Weinstein (2006, QJE 121(2):541-585): median elasticity of
    substitution sigma ~3.0-3.6 (the elasticities Ossa calibrates to).

Mapping (transparent, reduced-form):
  - Textbook optimal tariff ~ 1/(sigma-1). With sigma~3 -> ~0.5; Ossa's GE
    profit-shifting/political-economy motives push it to ~0.62. We set the
    optimal-tariff term beta/kappa to land Nash tariffs near 0.55-0.62.
  - Curvature kappa normalised to 1 (DWL convexity).
  - Asymmetry via terms-of-trade power beta: the US is the larger net-import
    market in the bilateral relationship, so it faces less elastic partner
    export supply and has MORE tariff leverage -> beta_US > beta_CN. (A modeling
    choice; the researcher can vary it. Direction follows standard optimal-
    tariff / market-power reasoning.)
  - delta (bilateral strategic coupling) small and positive: Ossa's optimal ~=
    Nash implies relatively flat reaction curves (mild strategic substitutes).
  - Welfare rescaled so the average per-country Nash loss = Ossa's 2.9% of GDP.
    (Scale is tariff- and delta*-invariant: it changes only the welfare axis.)

Payoffs (terms-of-trade gains are pure transfers, so cooperative optimum = free trade):
  W_US = beta_US*tUS - beta_CN*... no: each country's ToT gain is the partner's loss:
  W_US = beta_US*tUS  - beta_US-of-partner... see code for the exact transfer bookkeeping.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class AsymTradeGame:
    beta_US: float = 0.66     # US terms-of-trade extraction power (larger market)
    beta_CN: float = 0.56     # China terms-of-trade extraction power
    kappa_US: float = 1.0
    kappa_CN: float = 1.0
    delta: float = 0.10       # bilateral strategic coupling (mild substitutes)
    tau_max: float = 1.0
    w_scale: float = 1.0      # set by calibrate_to_ossa()

    # i's own tariff yields beta_i*tau_i, which is a transfer FROM the partner.
    def W_US(self, tUS, tCN):
        return self.w_scale*(self.beta_US*tUS - self.beta_CN*tCN
                             - 0.5*self.kappa_US*tUS**2 - self.delta*tUS*tCN)

    def W_CN(self, tUS, tCN):
        return self.w_scale*(self.beta_CN*tCN - self.beta_US*tUS
                             - 0.5*self.kappa_CN*tCN**2 - self.delta*tUS*tCN)

    def br_US(self, tCN):
        return float(np.clip((self.beta_US - self.delta*tCN)/self.kappa_US, 0, self.tau_max))

    def br_CN(self, tUS):
        return float(np.clip((self.beta_CN - self.delta*tUS)/self.kappa_CN, 0, self.tau_max))

    def nash(self, iters=500, tol=1e-12):
        tUS=tCN=0.0
        for _ in range(iters):
            nUS, nCN = self.br_US(tCN), self.br_CN(tUS)
            if abs(nUS-tUS)<tol and abs(nCN-tCN)<tol: tUS,tCN=nUS,nCN; break
            tUS,tCN=nUS,nCN
        return tUS,tCN

    def cooperative(self):
        return 0.0, 0.0  # joint welfare maximised at free trade (transfers cancel)

    def coordination_gap_pct(self):
        nUS,nCN=self.nash()
        loss_US=-self.W_US(nUS,nCN); loss_CN=-self.W_CN(nUS,nCN)
        joint_loss=loss_US+loss_CN
        return dict(nash=(nUS,nCN), loss_US=loss_US, loss_CN=loss_CN, joint_loss=joint_loss)

    def folk_delta_star(self):
        """critical discount per country; mutual cooperation needs max of the two."""
        out={}
        for who, beta, kappa, Wfun in [("US",self.beta_US,self.kappa_US,self.W_US),
                                       ("CN",self.beta_CN,self.kappa_CN,self.W_CN)]:
            nUS,nCN=self.nash()
            W_coop = 0.0
            # one-shot defection: i plays BR to a cooperating (free-trade) partner
            if who=="US": W_def=self.W_US(self.br_US(0.0),0.0); W_nash=self.W_US(nUS,nCN)
            else:         W_def=self.W_CN(0.0,self.br_CN(0.0)); W_nash=self.W_CN(nUS,nCN)
            ratio=(W_def-W_coop)/(W_coop-W_nash)
            out[who]=ratio/(1+ratio)
        out["binding"]=max(out["US"],out["CN"])
        return out

    def calibrate_to_ossa(self, target_pct=2.9):
        """scale welfare so AVERAGE per-country Nash loss = target_pct (% of GDP)."""
        self.w_scale=1.0
        g=self.coordination_gap_pct()
        avg_loss=0.5*(g["loss_US"]+g["loss_CN"])
        self.w_scale=target_pct/avg_loss
        return self.w_scale


if __name__=="__main__":
    g=AsymTradeGame(); g.calibrate_to_ossa(2.9)
    nUS,nCN=g.nash(); gap=g.coordination_gap_pct(); fd=g.folk_delta_star()
    print("="*60); print("CALIBRATED ASYMMETRIC US-CHINA TARIFF GAME"); print("="*60)
    print(f"beta_US={g.beta_US} beta_CN={g.beta_CN} kappa=1 delta={g.delta}")
    print(f"optimal tariff US (beta/kappa) = {g.beta_US/g.kappa_US:.1%}   [Ossa ~62%]")
    print(f"optimal tariff CN              = {g.beta_CN/g.kappa_CN:.1%}")
    print("-"*60)
    print(f"Nash tariffs:   US = {nUS:.1%}   China = {nCN:.1%}   [Ossa trade-war ~63%]")
    print(f"Cooperative:    free trade (0%, 0%)")
    print(f"Trade-war welfare loss: US={gap['loss_US']:.2f}%  CN={gap['loss_CN']:.2f}%  "
          f"avg={0.5*(gap['loss_US']+gap['loss_CN']):.2f}%  [Ossa ~2.9%]")
    print("-"*60)
    print(f"Folk-theorem critical discount:  US d*={fd['US']:.3f}  CN d*={fd['CN']:.3f}  "
          f"binding d*={fd['binding']:.3f}")
    # PD checks
    coopUS=g.W_US(0,0); defUS=g.W_US(g.br_US(0),0); nashUS=g.W_US(nUS,nCN)
    print(f"PD (US): defect>{'>' if defUS>coopUS else '<'}coop ({defUS:.2f} vs {coopUS:.2f}); "
          f"Nash<{'<' if nashUS<coopUS else '>'}coop ({nashUS:.2f} vs {coopUS:.2f})")
    print("="*60)
