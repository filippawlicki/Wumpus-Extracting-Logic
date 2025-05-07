% FACTS

% Dynamic properties of fields
:- dynamic wumpus/2.
:- dynamic pit/2.
:- dynamic gold/2.
:- dynamic agent/3.
:- dynamic start/2.
:- dynamic orientation/2.
:- dynamic kb/4.
:- dynamic result/2.
:- dynamic grid_size/1.
:- dynamic log/1.

adj(-1, 0).
adj(0, -1).
adj(0, 1).
adj(1, 0).
adj(1, 2).
adj(2, 1).
adj(2, 3).
adj(3, 2).
adj(3, 4).
adj(4, 3).

adjacent(X, Y, NX, NY) :-
	(NX is X, adj(Y, NY));
	(NY is Y, adj(X, NX)).

% PERCEPTIONS

% Stench occurs in the vicinity of the Wumpus
stench(X,Y) :-
    wumpus(WX, WY),
    adjacent(X, Y, WX, WY).

% Breeze in the vicinity of a pit
breeze(X,Y) :-
    pit(PX, PY),
    adjacent(X, Y, PX, PY).

% Glitter where the gold is
glitter(X,Y) :-
    gold(X,Y).


% Define cell_ahead to determine the cell in front of the agent
cell_ahead(X, Y, Dir, NX, NY) :-
    (Dir = north -> NX is X, NY is Y - 1;
     Dir = east -> NX is X + 1, NY is Y;
     Dir = south -> NX is X, NY is Y + 1;
     Dir = west -> NX is X - 1, NY is Y).

% Make decision with extended logic
make_decision(Action, S) :-
    agent(X, Y, S),
    orientation(Dir, S),
    cell_ahead(X, Y, Dir, NX, NY),
    kb(NX, NY, wall, IsWallAhead),
    (log(true) ->
	    nl, write('Agent: '), write((X, Y)),
	    write(', Dir: '), write(Dir),
	    write(', Cell: '), write((NX, NY)),
	    write(', Wall ahead: '), write(IsWallAhead),
	    nl
	;   true),
    (holding(gold, S), start(X, Y) ->
        Action = climb,
        (log(true) -> write('climbing out'), nl; true)
    ;   false, holding(gold, S) ->
            (previous_cell(NX, NY, S) ->
                Action = move,
                (log(true) -> write('moving to prev cell'), nl; true)
            ;   Action = turn_left,
                (log(true) -> write('turning to prev cell...'), nl; true)
            )
        ;   glitter(X, Y) -> 
                Action = grab,
                (log(true) -> write('grabbing gold'), nl; true)
            ;   wumpus_targeted(X, Y, Dir) ->
                    Action = shoot,
                    (log(true) -> write('shooting'), nl; true)
                ;   \+ valid_cell(NX, NY) ->
                        Action = turn_left,
                        (log(true) -> write('bumped, turning left'), nl; true)
                    ;   good_cell(NX, NY, S) -> 
                            Action = move,
                            (log(true) -> write('moving to good cell'), nl; true)
                        ;   once((adjacent(X, Y, GX, GY), good_cell(GX, GY, S))) -> 
                                (cell_right(X, Y, Dir, RX, RY),
                                good_cell(RX, RY, S) ->
                                    Action = turn_right,
                                    (log(true) -> write('turning to right good cell...'), nl; true)
                                ;   Action = turn_left,
                                    (log(true) -> write('turning to left good cell...'), nl; true)
                                )
                            ;   medium_cell(NX, NY) -> 
                                    Action = move,
                                    (log(true) -> write('moving to medium cell'), nl; true)
                                ;   once((adjacent(X, Y, MX, MY), medium_cell(MX, MY))) -> 
                                        (cell_right(X, Y, Dir, RX, RY),
                                        medium_cell(RX, RY) ->
                                            Action = turn_right,
                                            (log(true) -> write('turning to right medium cell...'), nl; true)
                                        ;   Action = turn_left,
                                            (log(true) -> write('turning to left medium cell...'), nl; true)
                                        )
                                    ;   risky_cell(NX, NY) -> 
                                            Action = move,
                                            (log(true) -> write('moving to risky cell'), nl; true)
                                        ;   once((adjacent(X, Y, RX, RY), risky_cell(RX, RY))) -> 
                                                (cell_right(X, Y, Dir, RX, RY),
                                                risky_cell(RX, RY) ->
                                                    Action = turn_right,
                                                    (log(true) -> write('turning to right good cell...'), nl; true)
                                                ;   Action = turn_left,
                                                    (log(true) -> write('turning to left good cell...'), nl; true)
                                                )
                                            ;   deadly_cell(NX, NY) -> 
                                                    Action = move,
                                                    (log(true) -> write('moving to deadly cell'), nl; true)
                                                ;   Action = turn_left,
                                                    (log(true) -> write('turning to deadly cell...'), nl; true)
    ).


% Define holding based on the result predicate
holding(gold, S) :-
    SPrev is S - 1,
    result(grab, SPrev).

holding(gold, S) :-
    S >= 1,
    SPrev is S - 1,
    holding(gold, SPrev).

has_shot(S) :-
	SPrev is S - 1,
	result(shoot, SPrev).

has_shot(S) :-
	S >= 1,
	SPrev is S - 1,
	has_shot(SPrev).


% Get the previous position of the agent based on the state S
previous_position(PrevX, PrevY, S) :-
    S > 0,
    SPrev is S - 1,
    agent(PrevX, PrevY, SPrev).

% Get the previous cell of the agent based on the state S
previous_cell(PrevX, PrevY, S) :-
    previous_position(PrevX, PrevY, S),
    agent(CurrX, CurrY, S),
    (PrevX \= CurrX; PrevY \= CurrY).

previous_cell(PrevX, PrevY, S) :-
    S > 0,
    SPrev is S - 1,
    previous_cell(PrevX, PrevY, SPrev).


% KNOWLEDGE BASE (KB)

% Initialize the knowledge base for all cells
initialize_kb(GridSize) :-
	% GridSizeM1 is GridSize - 1,
    forall(between(-1, GridSize, X),
           forall(between(-1, GridSize, Y),
                  (assert(kb(X, Y, visited, false)),
                   assert(kb(X, Y, stench, unknown)),
                   assert(kb(X, Y, breeze, unknown)),
                   assert(kb(X, Y, pit, unknown)),
                   assert(kb(X, Y, wumpus, unknown)),
                   assert(kb(X, Y, glitter, unknown)),
                   assert(kb(X, Y, wall, unknown))
                   ))).

% Update the knowledge base based on the agent's current position and perceptions
update_kb(X, Y, Perceptions, S) :-
    (log(true) -> write('Perceptions on cell ('), write((X, Y)), write('): '), write(Perceptions), nl; true),

    retract(kb(X, Y, visited, _)),
    assert(kb(X, Y, visited, true)),
    retract(kb(X, Y, wall, _)),
    assert(kb(X, Y, wall, false)),

    % Update stench and Wumpus knowledge
    (member(stench, Perceptions) -> 
        retract(kb(X, Y, stench, _)), assert(kb(X, Y, stench, true)),
        forall(adjacent(X, Y, NX, NY),
            (kb(NX, NY, wumpus, unknown) -> 
                retract(kb(NX, NY, wumpus, unknown)), assert(kb(NX, NY, wumpus, possible))
            ;   true))
    ;   retract(kb(X, Y, stench, _)), assert(kb(X, Y, stench, false)),
        forall(adjacent(X, Y, NX, NY),
            (retract(kb(NX, NY, wumpus, _)), assert(kb(NX, NY, wumpus, false))))
    ),

    % Deduce Wumpus location if only one possible remains
    findall((PX, PY), (adjacent(X, Y, PX, PY), kb(PX, PY, wumpus, possible)), PossibleWumpus),
    (length(PossibleWumpus, 1) ->
        [WumpusPos] = PossibleWumpus,
        WumpusPos = (WX, WY),
        retract(kb(WX, WY, wumpus, possible)), assert(kb(WX, WY, wumpus, true))
    ; true
    ),

    % Update breeze and Pit knowledge
    (member(breeze, Perceptions) -> 
        retract(kb(X, Y, breeze, _)), assert(kb(X, Y, breeze, true)),
        forall(adjacent(X, Y, NX, NY),
            (kb(NX, NY, pit, unknown) -> 
                retract(kb(NX, NY, pit, unknown)), assert(kb(NX, NY, pit, possible))
            ;   true))
    ;   retract(kb(X, Y, breeze, _)), assert(kb(X, Y, breeze, false)),
        forall(adjacent(X, Y, NX, NY),
            (retract(kb(NX, NY, pit, _)), assert(kb(NX, NY, pit, false))))
    ),

    % Deduce Pit location if only one possible remains
    findall((PX, PY), (adjacent(X, Y, PX, PY), kb(PX, PY, pit, possible)), PossiblePits),
    (length(PossiblePits, 1) ->
        [PitPos] = PossiblePits,
        PitPos = (PX, PY),
        retract(kb(PX, PY, pit, possible)), assert(kb(PX, PY, pit, true))
    ;   true
    ),

	(log(true) -> write('perceptions: '), write(Perceptions), nl; true),
    (member(bump, Perceptions) ->
    	(log(true) -> write('bump'), nl; true),
        orientation(Dir, S),
        cell_ahead(X, Y, Dir, BX, BY),
        retract(kb(BX, BY, wall, _)), assert(kb(BX, BY, wall, true))
    ;   true
    ),

    % Set current cell as false for Wumpus and Pit if not present
    (\+ member(stench, Perceptions) ->
            retract(kb(X, Y, wumpus, _)),
            assert(kb(X, Y, wumpus, false))
        ;   true),
    (\+ member(breeze, Perceptions) ->
            retract(kb(X, Y, pit, _)),
            assert(kb(X, Y, pit, false))
        ;   true),

    % Update glitter knowledge
    (member(glitter, Perceptions) -> 
        retract(kb(X, Y, glitter, _)), assert(kb(X, Y, glitter, true))
    ;   retract(kb(X, Y, glitter, _)), assert(kb(X, Y, glitter, false))
    ),

    (member(scream, Perceptions) ->
        (log(true) -> write('scream'), nl; true),
        retract(kb(WumpusX, WumpusY, wumpus, _)),
		assert(kb(WumpusX, WumpusY, wumpus, false))
	;   true
    ).

valid_cell(X, Y) :-
    \+ kb(X, Y, wall, true).

safe_cell(X, Y) :-
    kb(X, Y, pit, false),
    kb(X, Y, wumpus, false),
    valid_cell(X, Y).

visited(X, Y) :-
    kb(X, Y, visited, true).

good_cell(X, Y, S) :-
    safe_cell(X, Y),
    (holding(gold, S) ->
        visited(X, Y)
    ;   \+ visited(X, Y)
    ).

medium_cell(X, Y) :-
    safe_cell(X, Y),
    visited(X, Y).

risky_cell(X, Y) :-
    \+ deadly_cell(X, Y),
    \+ safe_cell(X, Y).

deadly_cell(X, Y) :-
    kb(X, Y, pit, true);
    kb(X, Y, wumpus, true).

cell_right(X, Y, Dir, NX, NY) :-
    (Dir = north -> NewDir = east;
    Dir = east -> NewDir = south;
    Dir = south -> NewDir = west;
    Dir = west -> NewDir = north
    ),
    cell_ahead(X, Y, NewDir, NX, NY).

wumpus_targeted(X, Y, Dir) :-
	kb(WX, WY, wumpus, true),
	DX is WX - X,
	DY is WY - Y,
	(Dir = north, DY < 0, DX = 0;
	Dir = east, DX > 0, DY = 0;
	Dir = south, DY > 0, DX = 0;
	Dir = west, DX < 0, DY = 0
	).
