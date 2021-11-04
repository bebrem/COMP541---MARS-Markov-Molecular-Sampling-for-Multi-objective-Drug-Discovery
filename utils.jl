
function standardize_smiles(mol)
    try
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
        return mol
    catch 
        return None
    end
end


function print_mols(run_dir, step, mols, scores, dicts)
    path = joinpath(run_dir,"mols.txt")
    open(path,"a") do f
        write(f,"molecules obtained at step $step")
        names = [i for i in keys(xlist[1])]
        nameslist = ""
        for i in range(1,length=length(names))
            nameslist = nameslist * " " * names[i] 
        end
        write(f,"score $(nameslist) smiles")
        for (i,mol) in enumerate(mols)
            try
                score = scores[i]
                mol = standardize_smiles(mol)
                mol = remove_all_hs(mol)
                smiles = get_smiles(mol)
                target_scores = [dicts[i][name] for name in names]
            catch
                score = 0.0 
                smiles = "[INVALID]"
                @assert False
                target_scores = [0.0 for name in names]
            end
            target_scores = [f * 1.0 for f in target_scores]
            
            target_score = ""
            for i in range(1,length=length(target_scores))
                target_score = target_score * " " * target_scores[i] 
            end           
            
            write(f,"$score $target_score $smiles")
        end    
    end
end