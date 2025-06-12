import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def kl_divergence(cov1,cov2):
    mu1 = mu2 = torch.zeros(1,cov1.shape[-1])
    mu1, cov1, mu2, cov2

    # Compute the KL divergence
    kl_div = 0.5 * (torch.trace(torch.inverse(cov2) @ cov1) +
                    (mu2 - mu1).T @ torch.inverse(cov2) @ (mu2 - mu1)  + torch.logdet(cov2) - torch.logdet(cov1))
    return kl_div.item()
def centeral_alignment_cortes(K):
  N = K.shape[-1]
  I = torch.ones(N)
  K = K  - 1/N*  I @ K - 1/N*  K @ I  + 1/(N^2) *  (torch.ones(1,N)@ K @ torch.ones(N,1)) * I
  return K
def centeral_alignment_cris(K):
  n = K.shape[-1]
  H = torch.eye(n) - torch.ones((n, n)) / n
  K = H @  K @ H
  return K


def my_loss(y_,output_, loss_type):
  y = y_.clone()
  output = output_.clone()
  # y = y[0,:,:]
  output = output[0,:,:]
  if loss_type == "MSE":
      loss = torch.sum(torch.square(output-y*2-1))
  elif loss_type == "centeral_alignment_cortes":
      y_c = centeral_alignment_cortes(y)
      output_c = centeral_alignment_cortes(output)
      loss = -torch.norm(y_c-output_c)/ (torch.norm(output_c)*torch.norm(y_c))
  elif loss_type == "centeral_alignment_cris":
      y_c = centeral_alignment_cris(y)
      output_c = centeral_alignment_cris(output)
      loss = -torch.norm(y_c-output_c)/ (torch.norm(output_c)*torch.norm(y_c))
  elif loss_type == "u_centeral_alignment_cortes":
      y_c = centeral_alignment_cortes(y)
      output_c = centeral_alignment_cortes(output)
      loss = -torch.trace(y_c.T @ output_c)/ (torch.trace(output_c.T@ output_c)*torch.trace(y_c.T @ y_c ))
  elif loss_type == "u_centeral_alignment_cris":
      y_c = centeral_alignment_cris(y)
      output_c = centeral_alignment_cris(output)
      loss = -torch.trace(y_c.T @output_c)/ (torch.trace(output_c.T @ output_c)*torch.trace(y_c.T @ y_c))
  elif loss_type == "FSM":
      output_P = output[y == 1]
      output_N = output[y == -1]
      s_p = torch.std(output_P)
      s_n = torch.std(output_N)
      m_p = torch.mean(output_P)
      m_n = torch.mean(output_N)
      loss =  (s_p+s_n)/torch.square(m_p-m_n+1e-9)
  elif loss_type == "structural_risk":
      V = torch.mul(output,y)
      P = V.T*V
      loss =  1/2*torch.sum(P)- torch.sum(output)

  elif loss_type == "structural_risk_logloss":
      V = torch.mul(output,y)
      P = (V.T*V)
      loss = torch.trace(P) + 0.001 *torch.sum(V)
  elif loss_type == "he":
      loss = torch.trace((output-y).T @ (output-y))
  return loss
  