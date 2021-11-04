using RDKitMinimalLib


function GetNumAtoms(mol)
    if typeof(mol) == RDKitMinimalLib.Mol
        mol = get_smiles(mol)
    end
    numAtoms = 0
    for char in mol
        if isletter(char)
            numAtoms += 1
        end
    end
    return numAtoms
end

function overlapMolecules(mol1,mol2)
    mol1_int = [parse(Int,m) for m in mol1]
    mol2_int = [parse(Int,m) for m in mol2]
    
    moland = [mol1_int[m] & mol2_int[m] for m in range(1,length=length(mol1_int))]
    m11 = sum(moland)
    
    molxorb = [moland[m] ⊻ mol2_int[m] for m in range(1,length=length(mol1_int))]
    m01 = sum(molxorb)

    molxora = [moland[m] ⊻ mol1_int[m] for m in range(1,length=length(mol1_int))]
    m10 = sum(molxora)
    
    div = m11 + m01 + m10
    
    if div == 0
        return 1
    else
        return m11 / div
    end
end


